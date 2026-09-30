"""Finite-size electrostatic charge correction (Rocklin et al. 2013).

Port of v1 ``abfe/charge_correction_generator.py``, kept as a library so
the analysis stage can drive it in-process instead of scraping its
stdout with regular expressions.  If you use this program please read
and cite:

    Calculating the binding free energies of charged species based on
    explicit-solvent simulations employing lattice-sum methods: An
    accurate correction scheme for electrostatic finite-size effects.
    Gabriel J. Rocklin, David L. Mobley, Ken A. Dill, and Philippe H.
    Hünenberger, J. Chem. Phys. 139, 184103 (2013).

Deviations from v1 (intentional):

* v1 scaled the lattice sum by ``li.a`` but divided some ddG terms by
  ``max(a, b, c)``; v2 uses ``a`` consistently (identical numbers for a
  rhombic dodecahedron box where a = b = c).
* a missing APBS ``.dx`` output raises a clear error instead of
  crashing on an undefined variable,
* results are returned as a :class:`ChargeSample` record.
"""

from __future__ import annotations

import argparse
import math
import os
import pathlib
import re
import subprocess
import sys
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from typing import Optional

import numpy

from .errors import PipelineError
from .topology import parse_index, parse_top
from .units import AVOGADRO, EPS0, KE

# Boltzmann constant in kJ/mol/K (== feplab.units.GAS_CONSTANT).
KB = 1.380649e-23 * AVOGADRO * 1.0e-3

# Atom classes in a TopologyInfo.
TOP_LIGAND = 0
TOP_RECEPTOR = 1
TOP_WATER = 2
TOP_IONS = 3
TOP_OTHER = 4

SOLVENT_NAME = "SOL"


def v_outer_product(p1, p2):
    x = p1[1] * p2[2] - p2[1] * p1[2]
    y = p1[2] * p2[0] - p2[2] * p1[0]
    z = p1[0] * p2[1] - p2[0] * p1[1]
    return numpy.array((x, y, z))


# Library of water models.  Dielectric constant values are from the OPC
# paper (Izadi 2014 JPCL), manually picked via Web Plot Digitizer; Izadi
# et al. calculated with an 8 A vdw cutoff and Berendsen barostat.
water_model_library = {
    "tip3p": {
        "name": "TIP3P Jorgensen 1983",
        "points": 3,
        "charges": [-0.834, 0.417, 0.417],
        "sigmas": [3.15061e-01, 0., 0.],
        "dielectric": [[261.52897202731066,263.5535182480578,265.30018557576113,267.1361711190857,268.97215666241027,270.4607935894302,272.14791544005277,273.735794828874,275.42291667949655,277.2489779766411,278.9956453043444,281.0201915250915,282.7668588527948,284.69657338782065,286.810989171199,288.9943233308282,291.1776574904574,293.4740177871381,295.701459707568,297.9289016279978,300.03504720622595,302.19357075040483,304.3686347048839,306.5960766253138,308.71876261384216,311.1502029279747,313.53202201120655,315.91384109443845,318.29566017767036,320.8759641845048,323.2577832677367,325.8380872745712,328.2199063578031,330.601725441035,333.18202944786947,335.56384853110137,337.9456676143333,340.3274866975652,342.70930578079697,345.09112486402887,347.4729439472608,349.95400549229396,352.23658211372447,354.81688612055905,357.19870520379095,359.58052428702274,362.06158583205604,364.44340491528783,367.38170347003154,372.5552050473186],
                       [127.03217908347989,125.075969652278,123.71163897205516,121.8858435029334,120.39826446294252,118.665621823878,116.87852060724155,115.1129161975414,113.3286812218298,111.28820430743768,109.92119846901832,108.03788709867148,106.71100863319985,104.92311123852873,103.46737931992493,102.10304863970208,100.71865427300537,99.1982726979858,97.54859181013464,95.88107653430671,94.15114090056008,92.49254281872055,90.94986825873002,89.22884981897178,87.51674857320192,86.15241789297907,85.04891513691648,83.91531685114307,82.84190962479127,81.9290118902304,80.96595493948486,80.0229616752132,79.03984103799378,78.02662487106358,77.06356792031804,76.00019253720316,74.91675346761443,73.8634099277365,72.8200982310955,71.76675469121756,70.69334746486577,69.63140520221332,68.5264693256883,67.47312578581037,66.44987777564323,65.38650239252836,64.57535621079923,63.38299998606665,62.14285714285714,60.51020408163265]]
    },
    "tips3p": {
        "name": "TIP3P CHARMM-modified",
        "points": 3,
        "charges": [-0.834, 0.417, 0.417],
        "sigmas": [0.315057422683, 0.0400013524445, 0.0400013524445],
        # dielectric copied from tip3p. TODO: simulate and test whether the value changes
        "dielectric": [[261.52897202731066,263.5535182480578,265.30018557576113,267.1361711190857,268.97215666241027,270.4607935894302,272.14791544005277,273.735794828874,275.42291667949655,277.2489779766411,278.9956453043444,281.0201915250915,282.7668588527948,284.69657338782065,286.810989171199,288.9943233308282,291.1776574904574,293.4740177871381,295.701459707568,297.9289016279978,300.03504720622595,302.19357075040483,304.3686347048839,306.5960766253138,308.71876261384216,311.1502029279747,313.53202201120655,315.91384109443845,318.29566017767036,320.8759641845048,323.2577832677367,325.8380872745712,328.2199063578031,330.601725441035,333.18202944786947,335.56384853110137,337.9456676143333,340.3274866975652,342.70930578079697,345.09112486402887,347.4729439472608,349.95400549229396,352.23658211372447,354.81688612055905,357.19870520379095,359.58052428702274,362.06158583205604,364.44340491528783,367.38170347003154,372.5552050473186],
                       [127.03217908347989,125.075969652278,123.71163897205516,121.8858435029334,120.39826446294252,118.665621823878,116.87852060724155,115.1129161975414,113.3286812218298,111.28820430743768,109.92119846901832,108.03788709867148,106.71100863319985,104.92311123852873,103.46737931992493,102.10304863970208,100.71865427300537,99.1982726979858,97.54859181013464,95.88107653430671,94.15114090056008,92.49254281872055,90.94986825873002,89.22884981897178,87.51674857320192,86.15241789297907,85.04891513691648,83.91531685114307,82.84190962479127,81.9290118902304,80.96595493948486,80.0229616752132,79.03984103799378,78.02662487106358,77.06356792031804,76.00019253720316,74.91675346761443,73.8634099277365,72.8200982310955,71.76675469121756,70.69334746486577,69.63140520221332,68.5264693256883,67.47312578581037,66.44987777564323,65.38650239252836,64.57535621079923,63.38299998606665,62.14285714285714,60.51020408163265]]
    },
    "spc": {
        "name": "SPC",
        "points": 3,
        "charges": [-0.82, 0.41, 0.41],
        "sigmas": [3.16557e-01, 0., 0.],
        # dielectric copied from SPC/E. TODO: revise the table, as SPC has lower eps than SPC/E.
        "dielectric": [[260.8342747947014,262.9955550739303,265.3112125159613,267.3181156323881,269.6337730744191,271.65170313104613,273.735794828874,276.11761391210587,278.1768949944834,280.5945516333658,282.55734698899204,284.90766052879496,287.034284710252,289.3877488043978,291.7979228767157,293.9245470581728,296.27486059797565,298.30822837088283,300.6553125924843,302.7145936748618,305.270087066246,307.47823184132557,309.61194477005415,311.90444563766476,313.96207267901235,316.628386819408,318.6374953238749,321.0248278772068,323.15854080593544,325.45529492190906,327.60460309463485,329.55967959212103,332.11241624845525,334.119319364882,336.50665191821406,338.6403648469426,340.9371189629162,343.0864271356421,345.0415036331282,347.5942402894624,349.6011434058893,351.9057739077201,354.12218888794973,356.1731997651772,358.59912660920963,360.56192196483585,362.9878488088684,365.03885968609586,367.2552746663255,369.5599051681563,371.52270052378265],
                       [81.6873356667952,80.99827976769276,80.24923547266845,79.58936311752798,78.80465004655014,78.17509615097018,77.37455506066294,76.71154142127658,76.20082940194183,75.52312265882459,75.07726295940536,74.43522499224167,73.91721708691641,73.32312351340455,72.61854938060803,72.06197750647587,71.44796500613282,70.92809259660999,70.26197820567765,69.70749087039992,68.91771121192868,68.2702013302721,67.66373080731208,67.03573742068009,66.47217076061419,65.6312793675095,65.09892288640295,64.36325438236122,63.783231446116744,63.13077078468092,62.49331880299697,62.00577122168204,61.20210911347887,60.64032589221064,59.89908414192617,59.32544510592338,58.647969978234414,58.039180405798845,57.52153729477311,56.73459492529817,56.17281170402994,55.69574182565135,55.31270781115029,54.96007332160963,54.49637923421362,54.17536025063178,53.72950055121254,53.35011447970672,52.953704674223076,52.534596556769,52.21357757318715]]
    },
    "spce": {
        "name": "SPC/E ",
        "points": 3,
        "charges": [-0.8476, 0.4238, 0.4238],
        "sigmas": [3.16557e-01, 0., 0.],
        "dielectric": [[260.8342747947014,262.9955550739303,265.3112125159613,267.3181156323881,269.6337730744191,271.65170313104613,273.735794828874,276.11761391210587,278.1768949944834,280.5945516333658,282.55734698899204,284.90766052879496,287.034284710252,289.3877488043978,291.7979228767157,293.9245470581728,296.27486059797565,298.30822837088283,300.6553125924843,302.7145936748618,305.270087066246,307.47823184132557,309.61194477005415,311.90444563766476,313.96207267901235,316.628386819408,318.6374953238749,321.0248278772068,323.15854080593544,325.45529492190906,327.60460309463485,329.55967959212103,332.11241624845525,334.119319364882,336.50665191821406,338.6403648469426,340.9371189629162,343.0864271356421,345.0415036331282,347.5942402894624,349.6011434058893,351.9057739077201,354.12218888794973,356.1731997651772,358.59912660920963,360.56192196483585,362.9878488088684,365.03885968609586,367.2552746663255,369.5599051681563,371.52270052378265],
                       [81.6873356667952,80.99827976769276,80.24923547266845,79.58936311752798,78.80465004655014,78.17509615097018,77.37455506066294,76.71154142127658,76.20082940194183,75.52312265882459,75.07726295940536,74.43522499224167,73.91721708691641,73.32312351340455,72.61854938060803,72.06197750647587,71.44796500613282,70.92809259660999,70.26197820567765,69.70749087039992,68.91771121192868,68.2702013302721,67.66373080731208,67.03573742068009,66.47217076061419,65.6312793675095,65.09892288640295,64.36325438236122,63.783231446116744,63.13077078468092,62.49331880299697,62.00577122168204,61.20210911347887,60.64032589221064,59.89908414192617,59.32544510592338,58.647969978234414,58.039180405798845,57.52153729477311,56.73459492529817,56.17281170402994,55.69574182565135,55.31270781115029,54.96007332160963,54.49637923421362,54.17536025063178,53.72950055121254,53.35011447970672,52.953704674223076,52.534596556769,52.21357757318715]]
    },
    # TODO TIP4P
    "tip4pew": {
        "name": "TIP4P/Ew ",
        "points": 4,
        "charges": [0., 0.52422, 0.52422, -1.04844],
        "sigmas": [3.16435e-01, 0.000, 0.000, 0.000],
        "dielectric": [[249.32214922574732,251.54518037009706,254.08578739221105,256.2124115736681,258.73600560233047,260.5762443940179,262.8588210154485,264.98260969799685,267.0617392727347,268.95230817005006,271.0314377447878,272.7929914417614,275.2953192286092,277.11003853011914,279.88882746055634,282.07216162018557,284.6855464476206,287.034284710252,289.26724010078186,291.99640780031837,294.1797419599476,297.3555007375901,299.5388348972193,302.9130785984645,305.0964127580937,308.2721715357362,310.52993754171644,313.0358097021999,315.16952263092844,317.60096294506104,319.6850546428889,321.9676312643195,324.2700563781102,327.3267242015912,329.51005836122033,332.68581713886283,334.75006034433056,337.38329366412574,339.03733469414783,342.01460854818777,344.197942707817,347.3737014854595,349.5570356450886,352.6137034685696,354.91612858236033,357.14908397289025,359.8782516724268,362.01196460115534,364.44340491528783,366.4530647667648,368.8100732345463,370.9437861632748,372.38280185939414],
                       [76.38322655170424,76.85891831828462,77.5067929942407,77.97789877534133,78.61993674250502,78.74490484682796,77.17191182727692,75.58286785854676,73.90353730068423,72.53318751451921,70.83379327018282,69.74032235735716,68.72567306996453,68.1486032304305,67.23236154812398,66.57025989448641,65.66070610767119,65.24240864421606,65.10561078189424,64.60766656304286,64.3766301733438,63.914557393945685,63.667713251267216,63.287111198762986,63.119305820981566,62.65966500358029,62.226471772894556,60.87217293590864,59.81882939603071,58.26754163730138,57.20720620668256,55.74316508458959,55.17408597733086,54.95520939761596,54.812939620801274,54.59771098408163,54.46784421345079,54.19765323520273,53.94351320693379,53.46198473156102,53.14582967197282,52.68618885457154,52.67159708259054,52.92695309225793,53.03639138211537,53.16771732994431,53.32822682173524,52.555774892491414,50.978951532795364,49.907368277941174,48.3597284622071,47.32918456604948,46.025044945248226]]
    },
    # TODO TIP4P/2005
    "opc": {
        "name": "OPC Izadi 2014",
        "points": 4,
        "charges": [0., 0.679142, 0.679142, -1.358284],
        "sigmas": [0.316655, 0.000, 0.000, 0.000],
        "dielectric": [[249.32214922574732,251.50548338537652,253.68881754500575,255.87215170463497,258.05548586426414,260.23882002389337,262.4221541835226,264.6054883431518,266.78882250278104,268.97215666241027,271.1554908220395,273.3388249816687,275.52215914129795,277.7054933009272,279.88882746055634,282.07216162018557,284.2554957798148,286.438829939444,288.6221640990732,290.8054982587024,292.98883241833164,295.17216657796087,297.3555007375901,299.5388348972193,301.72216905684854,303.90550321647777,306.088837376107,308.2721715357362,310.4555056953654,312.6388398549946,314.82217401462384,317.005508174253,319.18884233388223,321.37217649351146,323.5555106531407,325.7388448127699,327.92217897239914,330.10551313202836,332.2888472916576,334.4721814512868,336.65551561091604,338.83884977054527,341.0221839301745,343.2055180898036,345.38885224943283,347.57218640906206,349.7555205686913,351.9388547283205,354.12218888794973,356.30552304757896,358.4888572072082,360.6721913668374,362.85552552646664,365.03885968609586,367.2221938457251,369.4055280053543,371.58886216498354],
                       [96.44691302556967,95.54951904873859,94.703196273841,93.8568734989434,92.97407129409332,92.10586086122423,91.40545580613657,90.73423429501089,90.05571689789471,89.36990361478804,88.69138621767186,87.91072641668873,86.86741472004773,85.77303182147325,84.70053658087026,83.61344956828628,82.5190666697118,81.66544800882372,81.0452976996315,80.4470350484108,79.8195888532281,79.20673443002639,78.57928823484369,77.87888317975602,77.14929458070637,76.40511420967573,75.66822972463558,74.93134523959543,74.20905252653627,73.63997341927754,73.07819019800931,72.52370286273157,71.96921552745384,71.43661585014759,70.84564908491738,70.23279466171567,69.60534846653297,68.99978992932175,68.37234373413906,67.75219342494685,67.07367602783067,66.36597508675251,65.62909061071236,64.92868554662469,64.19180106158456,63.513283664468375,63.02445930310512,62.572114371694326,62.16354475622653,61.75497514075872,61.31722198132893,60.87946882189914,60.354165030583374,59.80697358129615,59.2816697899804,58.74907011267415,58.19458277739642]]
    },
    # TODO OPC3?
    "exp": {
        "name": "Experimeental",
        "points": 0,
        #TODO: replace with more accurate data
        "dielectric": [[249.12259651134121, 261.13913988537536, 273.08330617080946, 285.20842194520856, 298.0955594250039, 309.1384131884497, 325.09871951061825, 349.06294180059695, 373.0548869058],
                       [98.67904881166213, 92.97503637572734, 87.8880285489208, 83.19426503807563, 78.37368278908428, 74.5825141438473, 69.29620386629884, 62.022646411207546, 55.59522175294451]]
    }
}


@dataclass
class TopologyInfo:
    sigma: numpy.ndarray   # LJ sigma of each atom [nm]
    charge: numpy.ndarray  # charge of each atom [e]
    mtype: numpy.ndarray   # atom class: TOP_LIGAND / TOP_RECEPTOR / TOP_WATER / TOP_IONS / TOP_OTHER

    def guess_water_type(self) -> Optional[str]:
        """Guess the water model from the known libraries."""
        waters = self.mtype == TOP_WATER
        chargewaters = self.charge[waters]
        sigmawaters = self.sigma[waters]
        nwatatom: int = int(numpy.sum(waters))
        for watername, props in water_model_library.items():
            if watername == "exp":
                continue
            np = props["points"]
            if (np == 0 or nwatatom % np != 0
                    or numpy.any(chargewaters[np:] != chargewaters[:-np])):
                continue
            THRESHOLD = 1e-4
            dsigma = numpy.abs(sigmawaters[0:np] - numpy.array(props["sigmas"]))
            if numpy.any(dsigma > THRESHOLD):
                continue
            dcharge = numpy.abs(chargewaters[0:np] - numpy.array(props["charges"]))
            if numpy.any(dcharge > THRESHOLD):
                continue
            return watername
        return None


def build_topology_info(top, ndx, ligand_group, receptor_group) -> TopologyInfo:
    """Classify every system atom (port of v1 GMXTopParser.parse_top)."""
    gmxtop = parse_top(top)
    index = parse_index(ndx)

    atomtypetable = gmxtop.atomtypes
    moleculetypetable = gmxtop.moleculetypes

    natom = 0
    for (molname, nmol) in gmxtop.system:
        natom += len(moleculetypetable[molname]) * nmol
    moltypes = numpy.ones((natom,), dtype=int) * TOP_OTHER

    charges: list[float] = []
    sigmas: list[float] = []
    ipos = 0
    ligx = set(index[ligand_group])
    recx = set(index[receptor_group]) if receptor_group is not None else set()
    for (molname, nmol) in gmxtop.system:
        atoms = moleculetypetable[molname]
        if molname == SOLVENT_NAME:
            moltypes[ipos:ipos + nmol * len(atoms)] = TOP_WATER
        if len(atoms) == 1:
            charge = atoms[0].charge
            if charge is not None and round(charge) == charge and abs(charge) > 0:
                for i in range(nmol):
                    moltypes[ipos + i] = TOP_IONS
        for _i in range(nmol):
            for atom in atoms:
                (bondtype, _particle, _atomic_number, _mass, chargedefault,
                 sigma, _eps) = atomtypetable[atom.atomtype]
                charge = atom.charge if atom.charge is not None else chargedefault
                charges.append(charge)
                sigmas.append(sigma)
                if ipos in ligx:
                    moltypes[ipos] = TOP_LIGAND
                elif ipos in recx:
                    moltypes[ipos] = TOP_RECEPTOR
                ipos += 1
    return TopologyInfo(sigma=numpy.array(sigmas), charge=numpy.array(charges),
                        mtype=moltypes)


@dataclass
class LatticeInfo:
    a: float
    b: float
    c: float
    alpha: float
    beta: float
    gamma: float

    def get_unitcellvec(self):
        # For 90.0 degrees we want cos&sin EXACTLY 1&0 respectively, but
        # radian calculation may end up non-exact values.
        if self.alpha == 90.0:
            cosa, sina = 0.0, 1.0
        else:
            cosa, sina = math.cos(math.radians(self.alpha)), math.sin(math.radians(self.alpha))
        if self.beta == 90.0:
            cosb, sinb = 0.0, 1.0
        else:
            cosb, sinb = math.cos(math.radians(self.beta)), math.sin(math.radians(self.beta))
        if self.gamma == 90.0:
            cosc, sinc = 0.0, 1.0
        else:
            cosc, sinc = math.cos(math.radians(self.gamma)), math.sin(math.radians(self.gamma))

        units = []
        units.append(numpy.array((self.a, 0.0, 0.0)))
        units.append(numpy.array((self.b * cosc, self.b * sinc, 0.0)))
        units.append(numpy.array((self.c * cosb,
                                  self.c * (cosa - cosb * cosc) / sinc,
                                  self.c * math.sqrt(1 - cosb ** 2 - ((cosa - cosb * cosc) / sinc) ** 2))))
        return units

    def get_volume(self):
        units = self.get_unitcellvec()
        return float(v_outer_product(units[1], units[2]).dot(units[0]))

    @staticmethod
    def find_pdb_box(pdb) -> "LatticeInfo":
        """Find lattice information from the CRYST1 record of a PDB."""
        with open(pdb) as fin:
            for line in fin:
                if line.startswith("CRYST1"):
                    items = line.split()
                    return LatticeInfo(float(items[1]) / 10.0,  # angstrom -> nm
                                       float(items[2]) / 10.0,
                                       float(items[3]) / 10.0,
                                       float(items[4]), float(items[5]), float(items[6]))
        raise PipelineError('"CRYST1" entry was not found in the PDB')


class APBSCaller:
    """Calculate electrostatic potentials with APBS."""

    def __init__(self, topinfo: TopologyInfo, apbs_exe: str = "apbs",
                 apbs_template: Optional[pathlib.Path] = None):
        self.topinfo = topinfo
        self.apbs_exe = apbs_exe
        self.apbs_template = (pathlib.Path(apbs_template) if apbs_template is not None
                              else default_apbs_template())
        self.has_receptor = bool(numpy.any(topinfo.mtype == TOP_RECEPTOR))

    def setup(self, pdb) -> None:
        sigma = self.topinfo.sigma
        charge = self.topinfo.charge
        moltype = self.topinfo.mtype

        LatticeInfo.find_pdb_box(pdb)  # validates the CRYST1 record early

        self.save_pqr(pdb, "pro1lig0.pqr", sigma, charge, moltype, [TOP_RECEPTOR],
                      [TOP_LIGAND, TOP_RECEPTOR])
        self.save_pqr(pdb, "pro0lig1.pqr", sigma, charge, moltype, [TOP_LIGAND],
                      [TOP_LIGAND, TOP_RECEPTOR])
        self.save_pqr(pdb, "onlylig1.pqr", sigma, charge, moltype, [TOP_LIGAND],
                      [TOP_LIGAND])

    def calc(self, Lref, conc, epsS, temp, posion, negion) -> tuple[float, float, float]:
        self.save_inp(Lref, conc, epsS, temp, posion, negion)
        self.exec_pbs()

        if not self.has_receptor:
            B_HETP = 0.0
        else:
            B_HETP = self.average_pbs("pro1lig0") * KB * temp * Lref ** 3
        B_HETL = self.average_pbs("pro0lig1") * KB * temp * Lref ** 3
        B_HOML = self.average_pbs("onlylig1") * KB * temp * Lref ** 3
        return B_HETP, B_HETL, B_HOML

    def save_pqr(self, pdb, pqr, sigma, charge, moltype, enable_radii, show) -> None:
        # 6th root of 2.
        M_6_ROOT_2 = math.pow(2.0, 1.0 / 6.0)
        index = 0
        with open(pdb) as fin, open(pqr, "w") as fout:
            for line in fin:
                if not line.startswith("ATOM"):
                    continue
                rmin2 = sigma[index] * 10 * M_6_ROOT_2 / 2
                chg = charge[index] if moltype[index] in enable_radii else 0.0
                if moltype[index] in show:
                    # APBS specifies "whitespace-delimited" format, NOT PDB-like!
                    atom_number = index + 1
                    atom_name = line[12:16]
                    if atom_name == "    ":
                        atom_name = "DU  "
                    residue_name = line[17:20]
                    if residue_name == "   ":
                        residue_name = "DU "
                    chain_id = line[21]
                    if chain_id == " ":
                        chain_id = "X"
                    residue_number = line[22:26]
                    if residue_number == "    ":
                        residue_number = "X   "
                    x = line[30:38]
                    y = line[38:46]
                    z = line[46:54]
                    fout.write(f"ATOM {atom_number} {atom_name} {residue_name} {chain_id} "
                               f"{residue_number} {x} {y} {z} {chg:8.4f} {rmin2:8.4f}\n")
                index += 1

    def save_inp(self, Lref, conc, epsS, temp, pos_ion_radius, neg_ion_radius) -> None:
        dglen = 0.05  # [nm]
        dime = math.ceil(Lref / dglen)
        dime = math.ceil(dime / 32) * 32 + 1
        glen = (dime - 1) * (Lref / dime)  # [nm]

        with open(self.apbs_template) as ftemplate, open("apbs.in", "w") as fout:
            contents = ftemplate.read()
            fout.write(contents.format(dime=dime, glenaa=glen * 10.0, epsS=epsS,
                                       temp=temp, conc=conc,
                                       pos_ion_radius_aa=pos_ion_radius,
                                       neg_ion_radius_aa=neg_ion_radius))

    def exec_pbs(self) -> None:
        command = [self.apbs_exe, "apbs.in"]
        with open("apbs.log", "w") as fout:
            subprocess.check_call(command, stdout=fout)

    def average_pbs(self, dx_trunk) -> float:
        dxfile = pathlib.Path(dx_trunk + "-PE0.dx")
        if not dxfile.exists():
            raise PipelineError(f"APBS output {dxfile} not found (see apbs.log)")
        total = 0.0
        Nabc = None
        pat = re.compile(r"(^object 3 class array type double rank 0 items)\s+(\d+)")
        with open(dxfile) as fin:
            for line in fin:
                m = pat.match(line)
                if not m:
                    continue
                Nabc = int(m.group(2))
                count = 0
                for line in fin:
                    for item in line.split():
                        total += float(item)
                        count += 1
                        if count >= Nabc:
                            break
                    if count >= Nabc:
                        break
                break
        if Nabc is None:
            raise PipelineError(f"could not parse the APBS potential array in {dxfile}")
        avg = total / Nabc
        with open(dx_trunk + ".avg", "w") as fout:
            fout.write("%.6e\n" % avg)
        return avg


@dataclass
class ChargeSample:
    """One charge-correction sample (JSON-serializable via asdict)."""
    lattice: float   # box length a [nm], the scale of xi_LS
    xi_LS: float     # dimensionless lattice sum (scaled by lattice)
    volume: float    # [nm^3]
    eps_s: float     # relative dielectric permittivity of the solvent
    q_s: float       # net charge of the whole system [e]
    q_l: float       # net charge of the ligand [e]
    q_p: float       # net charge of the receptor [e]
    nsol: int        # number of solvent molecules
    gamma_s: float   # average quadrupole of a solvent molecule [e*nm^2]
    ip: float        # APBS interaction integral IP [kJ/mol]
    il: float        # IL [kJ/mol]
    l_ref: float     # APBS reference box length [nm]
    temp: float      # [K]

    def to_dict(self) -> dict:
        return asdict(self)


class RokhlinChargeCorrection:
    """All parameters and the correction-energy calculation."""

    def __init__(self, topinfo: TopologyInfo, pdb: str, temp: float,
                 apbs_exe: str = "apbs", apbs_template: Optional[pathlib.Path] = None):
        self.temp = temp
        self.pdbdata = self.load_pdb(pdb)
        self.topinfo = topinfo
        self.apbs = APBSCaller(topinfo, apbs_exe=apbs_exe, apbs_template=apbs_template)
        self.apbs.setup(pdb)
        self.summary = "chargecorr.log"

    @staticmethod
    def load_pdb(pdb):
        xs, ys, zs = [], [], []
        with open(pdb) as fin:
            for line in fin:
                if not line.startswith("ATOM  "):
                    continue
                xs.append(float(line[30:38].strip()) / 10.0)  # angstrom -> nm
                ys.append(float(line[38:46].strip()) / 10.0)
                zs.append(float(line[46:54].strip()) / 10.0)
        return (numpy.array(xs), numpy.array(ys), numpy.array(zs))

    def load_guess_initial_params(self, pdb) -> None:
        """Load inputs and guess the parameters."""
        print("Guessed parameters (some may later be overwritten by switches):")
        li = LatticeInfo.find_pdb_box(pdb)
        print("lengths[nm]:  %5.2f %5.2f %5.2f" % (li.a, li.b, li.c))
        print("angles[deg]:  %5.1f %5.1f %5.1f" % (li.alpha, li.beta, li.gamma))

        # v1 mixed max(a,b,c) and a here; v2 consistently uses a, which
        # xi_LS is scaled with.  (Identical for dodecahedron boxes.)
        self.lattice = li.a
        self.xi_LS, self.volume = self.calc_lattice_sum(li)
        print("V[nm^3]:             %10.3f" % self.volume)
        print("xi_LS:               %10.3f" % self.xi_LS)

        self.QL = self.calc_net_charge([TOP_LIGAND])
        self.QP = self.calc_net_charge([TOP_RECEPTOR])
        self.QS = self.calc_net_charge([TOP_LIGAND, TOP_RECEPTOR, TOP_WATER,
                                        TOP_IONS, TOP_OTHER])
        print("QP[e]:               %10g" % self.QP)
        print("QL[e]:               %10g" % self.QL)
        print("QS[e]:               %10g" % self.QS)

        self.nsol, self.gammaS = self.calc_quadrupole(pdb)
        print("gammaS[e*nm^2]:      %10.3e" % self.gammaS)
        print("Ns:                  %10d" % self.nsol)

        self.xi_CB = self.calc_cubic_coulomb()
        print("xi_CB:               %10.3f" % self.xi_CB)

        pdbcrds = self.pdbdata
        protmask = self.topinfo.mtype == TOP_RECEPTOR
        if numpy.sum(protmask) == 0:
            d = 0.0
        else:
            d = max(numpy.max(pdbcrds[0][protmask]) - numpy.min(pdbcrds[0][protmask]),
                    numpy.max(pdbcrds[1][protmask]) - numpy.min(pdbcrds[1][protmask]),
                    numpy.max(pdbcrds[2][protmask]) - numpy.min(pdbcrds[2][protmask]))
        self.Lref = d + 5 * 2  # 5 nm left-right. Will be ~ 15 nm

        self._get_ion_info(self.volume)
        self.epsS = self._get_water_eps()

        print("Lref[nm]:            %10.3f" % self.Lref)
        print("conc[mol/L]:         %10.3f (currently unused)" % self.conc)
        print("Pos ion Rmin/2[nm]:  %10.3f (currently unused)" % self.pos_ion_radius)
        print("Neg ion Rmin/2[nm]:  %10.3f (currently unused)" % self.neg_ion_radius)
        print("temp[K]:             %10.3f" % self.temp)
        print("epsS:                %10.3f" % self.epsS)
        print("")

    def _get_ion_info(self, vol: float) -> None:
        ionmask = (self.topinfo.mtype == TOP_IONS)
        sigma_ion = self.topinfo.sigma[ionmask]
        charge_ion = self.topinfo.charge[ionmask]
        pos_ion = charge_ion > 0
        neg_ion = charge_ion < 0
        assert int(numpy.sum(pos_ion)) + int(numpy.sum(neg_ion)) == int(numpy.sum(ionmask))
        nion_min: int = min(int(numpy.sum(pos_ion)), int(numpy.sum(neg_ion)))
        if nion_min == 0:
            self.conc = 0.0
            self.pos_ion_radius = 0.1369  # default Na
            self.neg_ion_radius = 0.2513  # default Cl
            return

        pos_sigmas = sigma_ion[pos_ion]
        neg_sigmas = sigma_ion[neg_ion]
        if numpy.any(pos_sigmas != pos_sigmas[-1]):
            sys.stderr.write("Warning: there are more than one positive ion species; "
                             "if it is a part of receptor, add to receptor group or specification\n")
        if numpy.any(neg_sigmas != neg_sigmas[-1]):
            sys.stderr.write("Warning: there are more than one negative ion species; "
                             "if it is a part of receptor, add to receptor group or specification\n")

        M_6_ROOT_2 = math.pow(2.0, 1.0 / 6.0)
        self.pos_ion_radius = pos_sigmas[-1] * 10 * M_6_ROOT_2 / 2
        self.neg_ion_radius = neg_sigmas[-1] * 10 * M_6_ROOT_2 / 2
        conc_nm3 = nion_min / vol  # 1/nm^3
        self.conc = conc_nm3 / AVOGADRO / (1e-9) ** 3 * 1e-3  # -> mol/L

    def _get_water_eps(self) -> float:
        watertype = self.topinfo.guess_water_type()
        if watertype is None:
            print("Warning: Unknown water model type. Static dielectric constant "
                  "(epsS) is guessed based on experimental water parameters")
            watertype = "exp"
        else:
            print(f"Water type:          {water_model_library[watertype]['name']}")
        de = water_model_library[watertype]["dielectric"]
        return float(numpy.interp(self.temp, de[0], de[1]))

    def calc_lattice_sum(self, lattice: LatticeInfo) -> tuple[float, float]:
        """Potential energy of a unit point charge in this lattice by Ewald sum."""
        units = lattice.get_unitcellvec()
        recps = [v_outer_product(units[1], units[2]),
                 v_outer_product(units[2], units[0]),
                 v_outer_product(units[0], units[1])]
        V = float(recps[0].dot(units[0]))
        recps = [r * (2 * math.pi / V) for r in recps]

        # The final value should be independent of this alpha
        alpha = math.sqrt(0.84 * math.pi / lattice.a ** 2)

        Ng = 3
        while True:
            sumg = 0.0
            fboundary = 0.0
            for iGa in range(-Ng, Ng + 1):
                for iGb in range(-Ng, Ng + 1):
                    for iGc in range(-Ng, Ng + 1):
                        G = recps[0] * iGa + recps[1] * iGb + recps[2] * iGc
                        GG = float(G.dot(G))
                        if GG > 0.0:
                            delta = (1.0 / V) * 4.0 * math.pi / GG * \
                                math.exp(-GG / (4.0 * alpha * alpha))
                        else:
                            delta = -(1.0 / V) * math.pi / (alpha * alpha)
                        sumg += delta
                        if iGa in (-Ng, Ng) or iGb in (-Ng, Ng) or iGc in (-Ng, Ng):
                            fboundary += abs(delta)
            if fboundary < 1e-5:
                break
            Ng += 1

        Nr = 3
        while True:
            sumr = 0.0
            fboundary = 0.0
            for ira in range(-Nr, Nr + 1):
                for irb in range(-Nr, Nr + 1):
                    for irc in range(-Nr, Nr + 1):
                        RL = units[0] * ira + units[1] * irb + units[2] * irc
                        drl = math.sqrt(float(RL.dot(RL)))
                        if drl > 0.0:
                            delta = math.erfc(alpha * drl) / drl
                        else:
                            delta = -alpha * (2.0 / math.sqrt(math.pi))
                        sumr += delta
                        if ira in (-Nr, Nr) or irb in (-Nr, Nr) or irc in (-Nr, Nr):
                            fboundary += abs(delta)
            if fboundary < 1e-5:
                break
            Nr += 1

        return (sumg + sumr) * lattice.a, V

    def calc_net_charge(self, mtypes) -> float:
        ret = 0.0
        for mtype in mtypes:
            ret += float(numpy.sum(self.topinfo.charge[self.topinfo.mtype == mtype]))
        return ret

    def calc_quadrupole(self, pdb) -> tuple[int, float]:
        """Averaged quadrupole of a solvent molecule."""
        water_type = 3  # 3-pt or 4-pt (TIP5P is currently unsupported)
        sigma_w = self.topinfo.sigma[self.topinfo.mtype == TOP_WATER]
        if len(sigma_w) % 3 == 0 and numpy.all(sigma_w[:-3] == sigma_w[3:]):
            water_type = 3
        elif len(sigma_w) % 4 == 0 and numpy.all(sigma_w[:-4] == sigma_w[4:]):
            water_type = 4
        else:
            raise PipelineError("Failed to find water molecules")

        xs, ys, zs = self.load_pdb(pdb)
        tot_quad = 0.0
        watpos = 0
        vdwpos = numpy.array((0., 0., 0.))
        for index in range(len(xs)):
            if self.topinfo.mtype[index] != TOP_WATER:
                continue
            xyz = numpy.array((xs[index], ys[index], zs[index]))
            if watpos % water_type == 0:
                # In all MD software the water molecules are ordered:
                # 3-pt: O(vdW,chg) H(chg) H(chg)  (CHARMM tip3p has vdw on H)
                # 4-pt: O(vdw), H(chg), H(chg), M(chg)
                # so the "vdW center" corresponds to the 0-th atom.
                vdwpos = xyz
            r = xyz - vdwpos
            tot_quad += self.topinfo.charge[index] * float(r.dot(r))
            watpos += 1
        nsol = watpos // water_type
        return (nsol, tot_quad / nsol)

    @staticmethod
    def calc_cubic_coulomb() -> float:
        return math.pi / 2 - 3.0 * math.log(2.0 + math.sqrt(3.0))

    def calc(self) -> float:
        """Correction energy of the solvation free energy, in kJ/mol."""
        # ddGnet
        ddGnet = -0.5 * KE * self.xi_LS * \
            ((self.QP + self.QL) ** 2 - self.QP ** 2) / self.lattice
        print("ddGnet[kJ/mol]:      %10.3f" % ddGnet)

        # ddGusv
        ddGusv = 0.5 * KE * self.xi_LS * (1.0 - 1.0 / self.epsS) * \
            ((self.QP + self.QL) ** 2 - self.QP ** 2) / self.lattice
        print("ddGusv[kJ/mol]:      %10.3f" % ddGusv)

        print("Running apbs...")
        print('Note: if you get "asc_getToken" error and/or "Vio_scanf", it is '
              "apbs's (harmless) issue, and you can ignore it.", file=sys.stderr)
        B_HETP, B_HETL, B_HOML = self.apbs.calc(
            self.Lref, self.conc, self.epsS, self.temp,
            self.pos_ion_radius, self.neg_ion_radius)
        print("Apbs finished.")

        B_HETQP = -KE / self.epsS * self.xi_CB * self.QP * self.Lref ** 2
        IP = B_HETP - B_HETQP
        print("  B_HETP/V[kJ/mol]:  %10.3f" % (B_HETP / self.volume))
        print("  B_HETQP/V[kJ/mol]: %10.3f" % (B_HETQP / self.volume))
        print("  IP/V[kJ/mol]:      %10.3f" % (IP / self.volume))

        B_HETQL = -KE / self.epsS * self.xi_CB * self.QL * self.Lref ** 2
        IL = B_HETL - B_HETQL
        print("  B_HETL/V[kJ/mol]:  %10.3f" % (B_HETL / self.volume))
        print("  B_HETQL/V[kJ/mol]: %10.3f" % (B_HETQL / self.volume))
        print("  IL/V[kJ/mol]:      %10.3f" % (IL / self.volume))

        ddGrip = ((IP + IL) * (self.QP + self.QL) - IP * self.QP) / self.volume
        print("ddGrip[kJ/mol]:      %10.3f" % ddGrip)

        # ddGemp
        B_HOMQL = -KE * self.xi_CB * self.QL * self.Lref ** 2
        print("  B_HOML/V[kJ/mol]:  %10.3f" % (B_HOML / self.volume))
        print("  B_HOMQL/V[kJ/mol]: %10.3f" % (B_HOMQL / self.volume))
        ILSLV = IL - (B_HOML - B_HOMQL)
        print("  ILSLV/V[kJ/mol]:   %10.3f" % (ILSLV / self.volume))

        if self.QL == 0.0:
            raise PipelineError("Error: RL becomes infinite due to QL=0.")
        if ILSLV / self.QL < 0.0:
            raise PipelineError("Error: RL becomes complex number due to ILSLV/QL<0.")
        RL = math.sqrt(ILSLV / (0.5 * KE * 4 * math.pi / 3 * (1 - 1 / self.epsS) * self.QL))
        print("  RL[nm]:            %10.3f" % RL)

        ddGemp = -0.5 * KE * 16 * math.pi ** 2 / 45 * (1 - 1 / self.epsS) * \
            ((self.QP + self.QL) ** 2 - self.QP ** 2) * RL ** 5 / self.volume ** 2
        print("ddGemp[kJ/mol]:      %10.3f" % ddGemp)

        # ddGana
        ddGana = ddGnet + ddGusv + ddGrip + ddGemp
        print("ddGana[kJ/mol]:      %10.3f" % ddGana)

        # ddGdsc
        ddGdsc = -self.gammaS * self.nsol / self.volume * self.QL / (6 * EPS0)
        print("ddGdsc[kJ/mol]:      %10.3f" % ddGdsc)

        ddG = ddGana + ddGdsc
        print("ddG[kJ/mol]:         %10.3f" % ddG)

        with open(self.summary, "w") as fout:
            fout.write("# ddGnet   ddGusv   ddGrip   ddGemp   ddGana   ddGdsc      ddG [kJ/mol]\n")
            fout.write("%8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f\n" %
                       (ddGnet, ddGusv, ddGrip, ddGemp, ddGana, ddGdsc, ddG))

        self._last_sample = ChargeSample(
            lattice=self.lattice, xi_LS=self.xi_LS, volume=self.volume,
            eps_s=self.epsS, q_s=self.QS, q_l=self.QL, q_p=self.QP,
            nsol=int(self.nsol), gamma_s=self.gammaS, ip=IP, il=IL,
            l_ref=self.Lref, temp=self.temp)
        return ddG

    @property
    def last_sample(self) -> ChargeSample:
        return self._last_sample


def default_apbs_template() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parent.parent / "apbs_input.template"


@contextmanager
def _chdir(path):
    path = pathlib.Path(path)
    path.mkdir(parents=True, exist_ok=True)
    prev = pathlib.Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev)


def run_sample(*, workdir, top, ndx, pdb, ligand_group="Ligand", receptor_group=None,
               temp: float, apbs_exe: str = "apbs", conc: float | None = None,
               lref: float | None = None, eps: float | None = None,
               apbs_template=None) -> ChargeSample:
    """Run one correction sample entirely inside ``workdir``.

    The APBS run produces fixed-name intermediates, so each sample is
    executed in its own directory.
    """
    with _chdir(workdir):
        topinfo = build_topology_info(str(top), str(ndx), ligand_group, receptor_group)
        corr = RokhlinChargeCorrection(topinfo, str(pdb), temp, apbs_exe=apbs_exe,
                                       apbs_template=apbs_template)
        corr.load_guess_initial_params(str(pdb))
        if conc is not None:
            print("Overwriting conc [mol/L] = %10.5f" % conc)
            corr.conc = conc
        if lref is not None:
            print("Overwriting Lref [nm]    = %10.5f" % lref)
            corr.Lref = lref
        if eps is not None:
            print("Overwriting epsS         = %10.5f" % eps)
            corr.epsS = eps
        corr.calc()
        return corr.last_sample


def main(argv=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--conc", "-c", type=float, default=None,
                        help="Salt concentration")
    parser.add_argument("--lref", default=None, type=float, dest="lref",
                        help="Force box size to be this number [nm]")
    parser.add_argument("--eps", type=float,
                        help="The permittivity of the solvent (default auto-detect)")
    parser.add_argument("--temp", type=float, default=300.0, help="Temperature (K)")
    parser.add_argument("--top", type=str,
                        help="GROMACS topology file (must be preprocessed)")
    parser.add_argument("--ndx", type=str,
                        help="GROMACS index file to specify which is ligand and which is receptor")
    parser.add_argument("--ligand-group", type=str, default="Ligand",
                        help="Ligand group name in the ndx file")
    parser.add_argument("--receptor-group", type=str, default=None,
                        help="Receptor group name in the ndx file")
    parser.add_argument("--pdb", type=str,
                        help="PDB file representing the input structure")
    parser.add_argument("--summary", type=str,
                        help="Output summary data to this file")
    args = parser.parse_args(argv)

    print("""Copyright 2021-2024 (C) Shun Sakruaba
This file is part of FEP-suite software.
You should have received a copy of the GNU General Public License along with this program. If not, see <https://www.gnu.org/licenses/>.

Also, if you use this program please read and cite:
Calculating the binding free energies of charged species based on explicit-solvent simulations employing lattice-sum methods: An accurate correction scheme for electrostatic finite-size effects.
Gabriel J. Rocklin, David L. Mobley, Ken A. Dill, and Philippe H. Hünenberger
The Journal of Chemical Physics, 139, 184103 (2013).""")

    topinfo = build_topology_info(args.top, args.ndx, args.ligand_group,
                                  args.receptor_group)
    corr = RokhlinChargeCorrection(topinfo, args.pdb, args.temp)
    corr.load_guess_initial_params(args.pdb)
    if args.conc is not None:
        print("Overwriting conc [mol/L] = %10.5f" % args.conc)
        corr.conc = args.conc
    if args.lref is not None:
        print("Overwriting Lref [nm]    = %10.5f" % args.lref)
        corr.Lref = args.lref
    if args.eps is not None:
        print("Overwriting epsS         = %10.5f" % args.eps)
        corr.epsS = args.eps
    if args.summary is not None:
        corr.summary = args.summary
    corr.calc()


if __name__ == "__main__":
    main()
