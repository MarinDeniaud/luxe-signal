import numpy as _np
import matplotlib.pyplot as _plt
import h5py as _h5
import jinja2 as _jj
import glob as _glob
import pandas as _pd
import pickle as _pk
from scipy.optimize import curve_fit
import os

yml_default_path = "../../00_ptarmigan/pickles/"
template_default_path = "../../00_ptarmigan/"
template_default_file_name = "luxe_template.yml"

# ============ GENERATE FILES ================ #


def GeneratePtarmiganFile(tag="000_luxe_default_jinja_10000", yml_path=yml_default_path,
                          templatefilename=template_default_file_name, templatefolder=template_default_path, **arg):
    ymlfilename = yml_path + tag + ".yml"
    paramdict = dict(tag=tag, **arg)
    env = _jj.Environment(loader=_jj.FileSystemLoader(templatefolder))
    template = env.get_template(templatefilename)
    f = open(ymlfilename, 'w')
    f.write(template.render(paramdict))
    f.close()

    return ymlfilename

# ============ RUN PTARMINGAN ================ #


def RunPtarmiganFile(file, printPtarmigan=False, removeYmlFile=False, mpi=False, mpi_nb=8):
    zsh = "zsh -l -c"
    if mpi:
        if mpi_nb > 8:
            raise ValueError("Number of cores requested '{}' is above 8".format(mpi_nb))
        command = "mpirun -n {} ptarmigan ".format(mpi_nb)
    else:
        command = "ptarmigan"
    if printPtarmigan:
        out = ""
    else:
        out = "&> /dev/null"
    os.system("{} '{} {}' {}".format(zsh, command, file, out))
    if removeYmlFile:
        os.system("rm {}".format(file))


def SetOffset(value, coord='X'):
    if coord == 'X':
        return [value, 0.0, 0.0]
    elif coord == 'Y':
        return [0.0, value, 0.0]
    elif coord == 'Z':
        return [0.0, 0.0, value]
    else:
        raise ValueError("Unknown coordinate '{}'".format(coord))


def GenerateRunAnalyse(data_dict,
                       getEnergyHist=False, printPtarmigan=False, removeYmlFile=False, removeH5File=False, mpi=False, mpi_nb=8, **arg):
    ymlfilename = GeneratePtarmiganFile(**arg)
    RunPtarmiganFile(ymlfilename, printPtarmigan=printPtarmigan, removeYmlFile=removeYmlFile, mpi=mpi, mpi_nb=mpi_nb)
    AnalyseFile(ymlfilename, data_dict, removeH5File=removeH5File, getEnergyHist=getEnergyHist)


def ScanNoJitter(tag="luxe_default_no_jitter_scan", ngenerate=10000, npoints=1000,
                 yml_path=yml_default_path, removeYmlFile=True, removeH5File=True, **arg):
    data_dict = {}
    printtag = "{}_ngen_{}_npts_{}".format(tag, ngenerate, npoints)
    for seed in range(npoints):
        fulltag = "{}_{}".format(seed, printtag)
        _printProgressBar(seed, npoints, prefix='Run no jitter scan : {}'.format(printtag), suffix='Complete', length=50)
        GenerateRunAnalyse(data_dict, tag=fulltag, seed=seed, ngenerate=ngenerate,
                           yml_path=yml_path, removeYmlFile=removeYmlFile, removeH5File=removeH5File, **arg)
    WritePicke(data_dict, "{}{}".format(yml_path, printtag))
    _printProgressBar(npoints, npoints, prefix='Run no jitter scan : {}'.format(printtag), suffix='Complete', length=50)


def ScanPositionJitter(tag="luxe_default_position_jitter_scan", ngenerate=10000, coord='X', mu=0, sigma=3e-6, npoints=1000,
                       yml_path=yml_default_path, removeYmlFile=True, removeH5File=True, **arg):
    data_dict = {}
    printtag = "{}_{}_ngen_{}_npts_{}".format(tag, coord, ngenerate, npoints)
    for seed in range(npoints):
        fulltag = "{}_{}".format(seed, printtag)
        position = _np.random.normal(mu, sigma)
        _printProgressBar(seed, npoints, prefix='Run position jitter scan : {}'.format(printtag), suffix='Complete', length=50)
        GenerateRunAnalyse(data_dict, tag=fulltag, seed=seed, ngenerate=ngenerate, offset=SetOffset(position, coord),
                           yml_path=yml_path, removeYmlFile=removeYmlFile, removeH5File=removeH5File, **arg)
    WritePicke(data_dict, "{}{}_mu_{:.2e}_sigma_{:.2e}".format(yml_path, printtag, mu, sigma))
    _printProgressBar(npoints, npoints, prefix='Run position jitter scan : {}'.format(printtag), suffix='Complete', length=50)


def ScanAngleJitter(tag="luxe_default_angle_jitter_scan", ngenerate=10000, mu=-17.2, sigma=6e-5, npoints=1000,
                    yml_path=yml_default_path, removeYmlFile=True, removeH5File=True, **arg):
    data_dict = {}
    printtag = "{}_ngen_{}_npts_{}".format(tag, ngenerate, npoints)
    for seed in range(npoints):
        fulltag = "{}_{}".format(seed, printtag)
        angle = _np.random.normal(mu, sigma)
        _printProgressBar(seed, npoints, prefix='Run angle jitter scan : {}'.format(printtag), suffix='Complete', length=50)
        GenerateRunAnalyse(data_dict, tag=fulltag, seed=seed, ngenerate=ngenerate, angle=angle,
                           yml_path=yml_path, removeYmlFile=removeYmlFile, removeH5File=removeH5File, **arg)
    WritePicke(data_dict, "{}{}_mu_{:.2e}_sigma_{:.2e}".format(yml_path, printtag, mu, sigma))
    _printProgressBar(npoints, npoints, prefix='Run angle jitter scan : {}'.format(printtag), suffix='Complete', length=50)


def ScanEnergyJitter(tag="luxe_default_energy_jitter_scan", ngenerate=10000, mu=16.5e9, sigma=1e6, npoints=1000,
                     yml_path=yml_default_path, removeYmlFile=True, removeH5File=True, **arg):
    data_dict = {}
    printtag = "{}_ngen_{}_npts_{}".format(tag, ngenerate, npoints)
    for seed in range(npoints):
        fulltag = "{}_{}".format(seed, printtag)
        energy = _np.random.normal(mu, sigma)
        _printProgressBar(seed, npoints, prefix='Run energy jitter scan : {}'.format(printtag), suffix='Complete', length=50)
        GenerateRunAnalyse(data_dict, tag=fulltag, seed=seed, ngenerate=ngenerate, E=energy,
                           yml_path=yml_path, removeYmlFile=removeYmlFile, removeH5File=removeH5File, **arg)
    WritePicke(data_dict, "{}{}_mu_{:.2e}_sigma_{:.2e}".format(yml_path, printtag, mu, sigma))
    _printProgressBar(npoints, npoints, prefix='Run energy jitter scan : {}'.format(printtag), suffix='Complete', length=50)


def ScanChargeJitter(tag="luxe_default_charge_jitter_scan", ngenerate=10000, mu=250e-12, sigma=1e-12, npoints=1000,
                     yml_path=yml_default_path, removeYmlFile=True, removeH5File=True, **arg):
    data_dict = {}
    printtag = "{}_ngen_{}_npts_{}".format(tag, ngenerate, npoints)
    for seed in range(npoints):
        fulltag = "{}_{}".format(seed, printtag)
        charge = _np.random.normal(mu, sigma)
        _printProgressBar(seed, npoints, prefix='Run charge jitter scan : {}'.format(printtag), suffix='Complete', length=50)
        GenerateRunAnalyse(data_dict, tag=fulltag, seed=seed, ngenerate=ngenerate, charge=charge,
                           yml_path=yml_path, removeYmlFile=removeYmlFile, removeH5File=removeH5File, **arg)
    WritePicke(data_dict, "{}{}_mu_{:.2e}_sigma_{:.2e}".format(yml_path, printtag, mu, sigma))
    _printProgressBar(npoints, npoints, prefix='Run charge jitter scan : {}'.format(printtag), suffix='Complete', length=50)


def ScanLengthJitter(tag="luxe_default_length_jitter_scan", ngenerate=10000, mu=12e-6, sigma=0.9e-6, npoints=1000,
                     yml_path=yml_default_path, removeYmlFile=True, removeH5File=True, **arg):
    data_dict = {}
    printtag = "{}_ngen_{}_npts_{}".format(tag, ngenerate, npoints)
    for seed in range(npoints):
        fulltag = "{}_{}".format(seed, printtag)
        length = _np.random.normal(mu, sigma)
        _printProgressBar(seed, npoints, prefix='Run length jitter scan : {}'.format(printtag), suffix='Complete', length=50)
        GenerateRunAnalyse(data_dict, tag=fulltag, seed=seed, ngenerate=ngenerate, length=length,
                           yml_path=yml_path, removeYmlFile=removeYmlFile, removeH5File=removeH5File, **arg)
    WritePicke(data_dict, "{}{}_mu_{:.2e}_sigma_{:.2e}".format(yml_path, printtag, mu, sigma))
    _printProgressBar(npoints, npoints, prefix='Run length jitter scan : {}'.format(printtag), suffix='Complete', length=50)


def ScanSizeJitter(tag="luxe_default_size_jitter_scan", ngenerate=10000, mu=5e-6, sigma=1e-6, npoints=1000,
                   yml_path=yml_default_path, removeYmlFile=True, removeH5File=True, **arg):
    data_dict = {}
    printtag = "{}_ngen_{}_npts_{}".format(tag, ngenerate, npoints)
    for seed in range(npoints):
        fulltag = "{}_{}".format(seed, printtag)
        size = _np.random.normal(mu, sigma)
        _printProgressBar(seed, npoints, prefix='Run size jitter scan : {}'.format(printtag), suffix='Complete', length=50)
        GenerateRunAnalyse(data_dict, tag=fulltag, seed=seed, ngenerate=ngenerate, radius=size,
                           yml_path=yml_path, removeYmlFile=removeYmlFile, removeH5File=removeH5File, **arg)
    WritePicke(data_dict, "{}{}_mu_{:.2e}_sigma_{:.2e}".format(yml_path, printtag, mu, sigma))
    _printProgressBar(npoints, npoints, prefix='Run size jitter scan : {}'.format(printtag), suffix='Complete', length=50)


def ScanPositionOffset(tag="luxe_default_position_offset_scan", seed=0, ngenerate=10000, coord='X', offset_list=[],
                       yml_path=yml_default_path, removeYmlFile=True, removeH5File=True, **arg):
    data_dict = {}
    noffset = len(offset_list)
    printtag = "{}_{}_ngen_{}".format(tag, coord, ngenerate)
    for i, offset in enumerate(offset_list):
        fulltag = "{}_{}_{}_{}_ngen_{}".format(seed, tag, coord, offset, ngenerate)
        _printProgressBar(i, noffset, prefix='Run position offset scan : {}'.format(printtag), suffix='Complete', length=50)
        GenerateRunAnalyse(data_dict, tag=fulltag, seed=seed, ngenerate=ngenerate, offset=SetOffset(offset, coord),
                           yml_path=yml_path, removeYmlFile=removeYmlFile, removeH5File=removeH5File, **arg)
        WritePicke(data_dict, "{}{}_{}".format(yml_path, seed, printtag))
    _printProgressBar(noffset, noffset, prefix='Run position offset scan : {}'.format(printtag), suffix='Complete', length=50)


def ScanRadiusOffset(tag="luxe_default_radius_offset_scan", seed=0, ngenerate=10000, radius_list=[],
                     yml_path=yml_default_path, removeYmlFile=True, removeH5File=True, **arg):
    data_dict = {}
    nradius = len(radius_list)
    printtag = "{}_ngen_{}".format(tag, ngenerate)
    for i, radius in enumerate(radius_list):
        fulltag = "{}_{}_{}_ngen_{}".format(seed, tag, radius, ngenerate)
        _printProgressBar(i, nradius, prefix='Run radius offset scan : {}'.format(printtag), suffix='Complete', length=50)
        GenerateRunAnalyse(data_dict, tag=fulltag, seed=seed, ngenerate=ngenerate, radius=radius,
                           yml_path=yml_path, removeYmlFile=removeYmlFile, removeH5File=removeH5File, **arg)
        WritePicke(data_dict, "{}{}_{}".format(yml_path, seed, printtag))
    _printProgressBar(nradius, nradius, prefix='Run radius offset scan : {}'.format(printtag), suffix='Complete', length=50)

# ============ ANALYSE FILES ================ #


def getLaserParam(file, param=None, returnUnit=False):
    data = _h5.File(file, 'r')
    if param is not None:
        value = data['config/laser/'+param][()]
        unit = data['config/laser/'+param].attrs['unit'].decode('utf-8')
    else:
        raise ValueError("Must specify 'param' parameter")
    data.close()
    if returnUnit:
        return value, unit
    return value


def getLaserA0(file, returnUnit=False):
    return getLaserParam(file, param='a0', returnUnit=returnUnit)


def getBeamParam(file, param=None, returnUnit=False):
    data = _h5.File(file, 'r')
    if param is not None:
        value = data['config/beam/'+param][()]
        unit = data['config/beam/'+param].attrs['unit'].decode('utf-8')
    else:
        raise ValueError("Must specify 'param' parameter")
    data.close()
    if returnUnit:
        return value, unit
    return value


def getBeamRadius(file, returnUnit=False):
    return getBeamParam(file, param='radius', returnUnit=returnUnit)


def getAllBeamOffset(file, returnUnit=False):
    return getBeamParam(file, param='offset', returnUnit=returnUnit)


def getBeamAngle(file, returnUnit=False):
    return getBeamParam(file, param='collision_angle', returnUnit=returnUnit)


def getBeamCharge(file, returnUnit=False):
    return getBeamParam(file, param='charge', returnUnit=returnUnit)


def getBeamEnergy(file, returnUnit=False):
    return getBeamParam(file, param='gamma', returnUnit=returnUnit)


def getBeamEnergyEV(file):
    E = getBeamParam(file, param='gamma', returnUnit=False)
    return E *  (9.1093837e-31 * 299792458 ** 2) / 1.602177e-19


def getBeamDE(file, returnUnit=False):
    return getBeamParam(file, param='sigma', returnUnit=returnUnit)


def getBeamLength(file, returnUnit=False):
    return getBeamParam(file, param='length', returnUnit=returnUnit)


def getOutput(file, part=None, param=None):
    data = _h5.File(file, 'r')
    if part is not None and param is not None:
        output = data['final-state'][part][param][:]
    else:
        raise ValueError("Must specify 'part' and 'param' parameters")
    data.close()
    return output


def getNumberParticles(file, part=None):
    momentum = getOutput(file, part=part, param='momentum')
    return momentum[:, 0].size


def getNumberParticlesWeighted(file, part=None):
    weight = getOutput(file, part=part, param='weight')
    return weight.sum()


def getEnergy(file, part=None):
    momentum = getOutput(file, part=part, param='momentum')
    return momentum[:, 0]


def fillDataDict(data_dict, file, getEnergyHist=False):
    offsets, offset_unit = getAllBeamOffset(file, returnUnit=True)
    radius,  radius_unit = getBeamRadius(file,    returnUnit=True)
    angle,   angle_unit  = getBeamAngle(file,     returnUnit=True)
    charge,  charge_unit = getBeamCharge(file,    returnUnit=True)
    length,  length_unit = getBeamLength(file,    returnUnit=True)
    energy,  energy_unit = getBeamEnergy(file,    returnUnit=True)

    data_dict["offset_unit"] = offset_unit
    data_dict["radius_unit"] = radius_unit
    data_dict["angle_unit"] = angle_unit
    data_dict["charge_unit"] = charge_unit
    data_dict["length_unit"] = length_unit
    data_dict["energy_unit"] = energy_unit

    try:
        data_dict["BeamOffsetX"].append(offsets[0])
        data_dict["BeamOffsetY"].append(offsets[1])
        data_dict["BeamOffsetZ"].append(offsets[2])
        data_dict["BeamRadius"].append(radius)
        data_dict["BeamAngle"].append(angle)
        data_dict["BeamCharge"].append(charge)
        data_dict["BeamLength"].append(length)
        data_dict["BeamEnergy"].append(energy)

        data_dict["electron"]["Nbparticle"].append(getNumberParticlesWeighted(file, part='electron'))
        data_dict["photon"]["Nbparticle"].append(getNumberParticlesWeighted(file, part='photon'))
        data_dict["positron"]["Nbparticle"].append(getNumberParticlesWeighted(file, part='positron'))
        if getEnergyHist:
            data_dict["electron"]["Energy"].append(getEnergy(file, part='electron').tolist())
            data_dict["photon"]["Energy"].append(getEnergy(file, part='photon').tolist())
            data_dict["positron"]["Energy"].append(getEnergy(file, part='positron').tolist())

    except KeyError:
        data_dict["BeamOffsetX"] = [offsets[0]]
        data_dict["BeamOffsetY"] = [offsets[1]]
        data_dict["BeamOffsetZ"] = [offsets[2]]
        data_dict["BeamRadius"] = [radius]
        data_dict["BeamAngle"] = [angle]
        data_dict["BeamCharge"] = [charge]
        data_dict["BeamLength"] = [length]
        data_dict["BeamEnergy"] = [energy]

        data_dict["electron"] = {}
        data_dict["photon"] = {}
        data_dict["positron"] = {}

        data_dict["electron"]["Nbparticle"] = [getNumberParticlesWeighted(file, part='electron')]
        data_dict["photon"]["Nbparticle"] = [getNumberParticlesWeighted(file, part='photon')]
        data_dict["positron"]["Nbparticle"] = [getNumberParticlesWeighted(file, part='positron')]
        if getEnergyHist:
            data_dict["electron"]["Energy"] = [getEnergy(file, part='electron').tolist()]
            data_dict["photon"]["Energy"] = [getEnergy(file, part='photon').tolist()]
            data_dict["positron"]["Energy"] = [getEnergy(file, part='positron').tolist()]


def AnalyseFile(ymlfilename, data_dict, removeH5File=False, getEnergyHist=False):
    h5filename = _glob.glob('{}*'.format(".." + ymlfilename.strip(".yml")))[0]
    fillDataDict(data_dict, h5filename, getEnergyHist=getEnergyHist)
    if removeH5File:
        os.system("rm {}".format(h5filename))


def calcPercentile(Npart, cumsum, value):
    Npart1 = Npart[cumsum < value][-1]
    Npart2 = Npart[cumsum > value][0]
    cumsum1 = cumsum[cumsum < value][-1]
    cumsum2 = cumsum[cumsum > value][0]

    slope = (cumsum2 - cumsum1) / (Npart2 - Npart1)
    intercept = cumsum1 - slope * Npart1

    return (value - intercept)/slope


def calcCumulativeLength(data, nbins=30, value=0.68):
    Hist = _np.histogram(data, bins=nbins)
    cumsum = _np.cumsum(Hist[0]/Hist[0].sum())
    Npart = (Hist[1][:-1] + Hist[1][1:]) / 2
    upper_cl = calcPercentile(Npart, cumsum, value)
    lower_cl = calcPercentile(Npart, cumsum, 1 - value)
    mode = calcPercentile(Npart, cumsum, 0.5)
    return upper_cl, lower_cl, mode


def calcVariation(upper_cl, lower_cl, mode):
    return _np.abs(upper_cl - lower_cl) / mode * 100
    # return _np.abs(upper_cl - lower_cl)

# ============ WRITE/READ ================ #


def WritePicke(data_dict, outputfilename):
    with open('{}.pickle'.format(outputfilename), 'wb') as file:
        _pk.dump(data_dict, file, protocol=_pk.HIGHEST_PROTOCOL)


def ReadPicke(inputfilename):
    with open('{}'.format(inputfilename), 'rb') as file:
        data_dict = _pk.load(file)
    return data_dict

# ============ PLOTS ================ #


def plotEnergyCurvesforEachA0(regex, part='photon'):
    filenames = _glob.glob(regex)
    Npart = []
    a0 = []
    E = []
    for filename in filenames:
        Npart.append(getNumberParticlesWeighted(filename, part=part))
        a0.append(getLaserA0(filename))
        E.append(getBeamEnergyEV(filename))
    df = _pd.DataFrame(_np.array([Npart, a0, E]).transpose(), columns=['Npart', 'a0', 'E'])
    for a0 in _np.sort(df.a0.unique()):
        df_reduced = df[df.a0 == a0]
        df_reduced = df_reduced.sort_values(by='E')
        _plt.plot(df_reduced.E.values, df_reduced.Npart.values, ls='', marker='+', markersize=12,
                  label="a0={}".format(a0))
    _plt.ylabel('Number of {}'.format(part))
    _plt.xlabel("Energy (eV)")
    _plt.legend()


def plotA0CurvesforEachEnergy(regex, part='photon'):
    filenames = _glob.glob(regex)
    Npart = []
    a0 = []
    E = []
    for filename in filenames:
        Npart.append(getNumberParticlesWeighted(filename, part=part))
        a0.append(getLaserA0(filename))
        E.append(getBeamEnergyEV(filename))
    df = _pd.DataFrame(_np.array([Npart, a0, E]).transpose(), columns=['Npart', 'a0', 'E'])
    for energy in _np.sort(df.E.unique()):
        df_reduced = df[df.E == energy]
        df_reduced = df_reduced.sort_values(by='a0')
        _plt.plot(df_reduced.a0.values, df_reduced.Npart.values, ls='', marker='+', markersize=12,
                  label="{:.2e} eV".format(energy))
    _plt.ylabel('Number of {}'.format(part))
    _plt.xlabel("a0")
    _plt.legend()


def plotAllA0EnergyCurves(regex, figsize=[13, 5]):
    _plt.rcParams['font.size'] = 15
    fig = _plt.figure(figsize=figsize)
    _plt.subplot(1, 3, 1)
    plotA0CurvesforEachEnergy(regex, part='photon')
    _plt.subplot(1, 3, 2)
    plotA0CurvesforEachEnergy(regex, part='positron')
    _plt.subplot(1, 3, 3)
    plotA0CurvesforEachEnergy(regex, part='electron')
    fig.tight_layout()


def plotCompare(file1, file2, part='photon', param='E'):
    data1 = _h5.File(file1, 'r')
    data2 = _h5.File(file2, 'r')
    if param == "E":
        set1 = data1['final-state'][part]['momentum'][:][:, 0]
        set2 = data2['final-state'][part]['momentum'][:][:, 0]
        xlabel = 'energy [GeV]'
    elif param == "X":
        set1 = data1['final-state'][part]['position'][:][:, 1]
        set2 = data2['final-state'][part]['position'][:][:, 1]
        xlabel = 'X [m]'
    else:
        raise ValueError("Unknown input parameter : {}".format(param))

    data1.close()
    data2.close()

    n1 = set1.size
    n2 = set2.size

    _plt.figure(figsize=[12, 4])
    _plt.subplot(1, 2, 1)
    _plt.hist(set1, bins=50, label="tt {} : {}".format(part, n1))
    _plt.xlabel(xlabel)
    _plt.legend()
    _plt.subplot(1, 2, 2)
    _plt.hist(set2, bins=50, label="tt {} : {}".format(part, n2))
    _plt.xlabel(xlabel)
    _plt.legend()

    print("{:.3f}% difference".format(_np.abs(n1 - n2) / max(n1, n2) * 100))


def plotCurve(inputfilename, X='OffsetX', Y='Nbphoton', unit='offset_unit', figsize=[8, 5]):
    data_dict = ReadPicke(inputfilename)

    _plt.rcParams['font.size'] = 15
    _plt.figure(figsize=figsize)
    _plt.plot(data_dict[X], data_dict[Y], ls='', marker='+', markersize=12)
    _plt.ylabel("{}".format(Y))
    _plt.xlabel("{} [{}]".format(X, data_dict[unit]))
    _plt.ticklabel_format(axis="both", style='sci', scilimits=(0, 0))


def plotHist(inputfilename, X='Nbphoton', nbins=20, figsize=[8, 5]):
    data_dict = ReadPicke(inputfilename)

    _plt.rcParams['font.size'] = 15
    _plt.figure(figsize=figsize)
    _plt.hist(data_dict[X], histtype='step', bins=nbins, label="std : {:.2e}".format(_np.std(data_dict[X])))
    _plt.xlabel("{}".format(X))
    _plt.ticklabel_format(axis="both", style='sci', scilimits=(0, 0))
    _plt.legend()


def plotMultipleHist(inputfilename, X=['Nbphoton'], nbins=20, figsize=[8, 5]):
    data_dict = ReadPicke(inputfilename)

    _plt.rcParams['font.size'] = 15
    _plt.figure(figsize=figsize)
    for x in X:
        _plt.hist(data_dict[x], histtype='step', bins=nbins, label="std_{} : {:.2e}".format(x, _np.std(data_dict[x])))
    _plt.xlabel("Nbparticles")
    _plt.ticklabel_format(axis="both", style='sci', scilimits=(0, 0))
    _plt.legend()


def plotAllHist(inputfilename, param='Energy', unit='GeV', nbins=20, figsize=[8, 5]):
    data_dict = ReadPicke(inputfilename)

    _plt.rcParams['font.size'] = 15
    _plt.figure(figsize=figsize)
    _plt.hist(sum(data_dict['photon'][param], []), histtype='step', bins=nbins, label="Photons")
    _plt.hist(sum(data_dict['positron'][param], []), histtype='step', bins=nbins, label="Positrons")
    _plt.hist(sum(data_dict['electron'][param], []), histtype='step', bins=nbins, label="Electrons")
    _plt.xlabel("{} [{}]".format(param, unit))
    _plt.ticklabel_format(axis="both", style='sci', scilimits=(0, 0))
    _plt.legend()


def plotCumulativeLength(inputfilename, X='photon', nbins=30, value=0.68, cumulative=True):
    data_dict = ReadPicke(inputfilename)
    upper_cl, lower_cl, mode = calcCumulativeLength(data_dict[X]['Nbparticle'])
    variation = calcVariation(upper_cl, lower_cl, mode)

    _plt.hist(data_dict[X]['Nbparticle'], bins=nbins, histtype='step', label="Signal {}".format(X))
    if cumulative:
        _plt.hist(data_dict[X]['Nbparticle'], bins=nbins, histtype='step', color='k', alpha=0.3, cumulative=True, label="Cumulative")

    _plt.xlabel(X)
    _plt.axvline(upper_cl, ls='--', color='k', alpha=0.3, label="CL at {} : {:.2f} %".format(value, variation))
    _plt.axvline(lower_cl, ls='--', color='k', alpha=0.3)
    _plt.legend()


def plotAllParticlesCL(inputfilename, nbins=30, value=0.68, cumulative=True, figsize=[8, 5]):
    _plt.rcParams['font.size'] = 15
    _plt.figure(figsize=figsize)
    _plt.subplot(1, 3, 1)
    plotCumulativeLength(inputfilename, X='photon', nbins=nbins, value=value, cumulative=cumulative)
    _plt.subplot(1, 3, 2)
    plotCumulativeLength(inputfilename, X='positron', nbins=nbins, value=value, cumulative=cumulative)
    _plt.subplot(1, 3, 3)
    plotCumulativeLength(inputfilename, X='electron', nbins=nbins, value=value, cumulative=cumulative)


def plotResolutionOnResolution(regex, nbins=30, value=0.68, figsize=[8, 5], fitparam=[1e10, 1]):
    filenames = _glob.glob(regex)
    if not filenames:
        raise IOError("No files found with regex : {}".format(regex))
    Var_BPM = []
    Var_IP_Photon = []
    Var_IP_Positron = []
    for filename in filenames:
        Var_BPM.append(float(filename.strip('.pickle').split('_')[-1]))

        data_dict = ReadPicke(filename)
        upper_cl_photon, lower_cl_photon, mode_photon = calcCumulativeLength(data_dict['Nbphoton'], nbins=nbins, value=value)
        upper_cl_positron, lower_cl_positron, mode_positron = calcCumulativeLength(data_dict['Nbpositron'], nbins=nbins, value=value)
        Var_IP_Photon.append(calcVariation(upper_cl_photon, lower_cl_photon, mode_photon))
        Var_IP_Positron.append(calcVariation(upper_cl_positron, lower_cl_positron, mode_positron))

    def hyper(x, a, c):
        return _np.sqrt(pow(a, 2) * pow(x, 2) + pow(c, 2))

    def poly(x, a, c):
        return a * pow(x, 2) + c

    _plt.rcParams['font.size'] = 15
    fig, ax1 = _plt.subplots(figsize=figsize)
    ax2 = ax1.twinx()

    params, covariance = curve_fit(poly, Var_BPM, Var_IP_Photon, p0=fitparam)
    a_fit, c_fit = params
    x_fit = _np.linspace(min(Var_BPM), max(Var_BPM), 100)
    y_fit = poly(x_fit, a_fit, c_fit)
    line0, = ax1.plot(x_fit, y_fit, label="Polynomial fit", color='C0')

    line1, = ax1.plot(Var_BPM, Var_IP_Photon, ls='', marker='+', markersize=12, color='C0', label='Photons')
    ax1.set_ylabel(r"$\sigma_{Photons}$ [%]", color='C0')
    ax1.set_xlabel(r"$\sigma_{Jitter}$ [m]")

    line2, = ax2.plot(Var_BPM, Var_IP_Positron, ls='', marker='x', markersize=12, color='C1', label='Positrons')
    ax2.set_ylabel(r"$\sigma_{Positrons}$ [%]", color='C1')

    _plt.ticklabel_format(axis="x", style='sci', scilimits=(0, 0))
    lines = [line0, line1, line2]
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels)
    fig.tight_layout()


def getSignalJitterArray(filenames, part='photon', nbins=30, value=0.68):
    Var_BPM = []
    Var_IP  = []
    for filename in filenames:
        Var_BPM.append(float(filename.strip('.pickle').split('_')[-1]))

        data_dict = ReadPicke(filename)
        upper_cl, lower_cl, mode = calcCumulativeLength(data_dict[part]['Nbparticle'], nbins=nbins, value=value)
        Var_IP.append(calcVariation(upper_cl, lower_cl, mode))

    return Var_BPM, Var_IP


def fitJitter(func, Var_BPM, Var_IP, fitparam=[]):
    params, covariance = curve_fit(func, Var_BPM, Var_IP, p0=fitparam)
    a_fit, c_fit = params
    x_fit = _np.linspace(min(Var_BPM), max(Var_BPM), 100)
    y_fit = func(x_fit, a_fit, c_fit)
    return x_fit, y_fit, a_fit, c_fit


def plotXYResolutionOnResolution(regex_x, regex_y=None, labelX="X", labelY="Y", labelXY="X/Y", unit="m",
                                 part='photon', nbins=30, value=0.68,
                                 fit=False, fitparam=[1e10, 1], mark_x=None, mark_y=None):
    filenames_x = _glob.glob(regex_x)
    if not filenames_x:
        raise IOError("No files found with regex")

    Var_BPM_X, Var_IP_X = getSignalJitterArray(filenames_x, part=part, nbins=nbins, value=value)

    def hyper(x, a, c):
        return _np.sqrt(pow(a, 2) * pow(x, 2) + pow(c, 2))

    def poly(x, a, c):
        return a * pow(x, 2) + c

    if fit:
        JitterX_fit_x, JitterX_fit_y, JitterX_fit_a, JitterX_fit_c = fitJitter(poly, Var_BPM_X, Var_IP_X, fitparam=fitparam)
        _plt.plot(JitterX_fit_x, JitterX_fit_y, label="fit", color='C0')
        if mark_x is not None:
            _plt.axvline(mark_x, ls='--', color='C0', alpha=0.5)
            _plt.axhline(poly(mark_x, JitterX_fit_a, JitterX_fit_c), ls=':', color='C0', alpha=0.5)
    _plt.plot(Var_BPM_X, Var_IP_X, ls='', marker='+', markersize=12, color='C0', label='data {}'.format(labelX))

    if regex_y is not None:
        filenames_y = _glob.glob(regex_y)
        Var_BPM_Y, Var_IP_Y = getSignalJitterArray(filenames_y, part=part, nbins=nbins, value=value)
        if fit:
            JitterY_fit_x, JitterY_fit_y, JitterY_fit_a, JitterY_fit_c = fitJitter(poly, Var_BPM_Y, Var_IP_Y, fitparam=fitparam)
            _plt.plot(JitterY_fit_x, JitterY_fit_y, label="fit", color='C1')
            if mark_y is not None:
                _plt.axvline(mark_y, ls='--', color='C1', alpha=0.5)
                _plt.axhline(poly(mark_y, JitterY_fit_a, JitterY_fit_c), ls=':', color='C1', alpha=0.5)
        _plt.plot(Var_BPM_Y, Var_IP_Y, ls='', marker='x', markersize=12, color='C1', label='data {}'.format(labelY))

    _plt.ylabel(r"$\Sigma_{{{part}}}$ [%]".format(part=part))
    _plt.xlabel(r"$\sigma_{{J, {param}}}$ [{unit}]".format(param=labelXY, unit=unit))
    _plt.ticklabel_format(axis="x", style='sci', scilimits=(0, 0))
    _plt.legend()


def plotAllResOnRes(regex_x, regex_y=None, nbins=30, value=0.68, labelX="X", labelY="Y", labelXY="X/Y", unit='m',
                    fit=False, fitparams=[[1e10, 1], [1e10, 1], [1e10, 1]],
                    mark_x=None, mark_y=None, figsize=[13, 5]):
    _plt.rcParams['font.size'] = 15
    fig = _plt.figure(figsize=figsize)

    _plt.subplot(1, 3, 1)
    plotXYResolutionOnResolution(regex_x, regex_y, labelX=labelX, labelY=labelY, labelXY=labelXY, unit=unit,
                                 part='photon', nbins=nbins, value=value,
                                 fit=fit, fitparam=fitparams[0], mark_x=mark_x, mark_y=mark_y)
    _plt.subplot(1, 3, 2)
    plotXYResolutionOnResolution(regex_x, regex_y, labelX=labelX, labelY=labelY, labelXY=labelXY, unit=unit,
                                 part='positron', nbins=nbins, value=value,
                                 fit=fit, fitparam=fitparams[1], mark_x=mark_x, mark_y=mark_y)
    _plt.subplot(1, 3, 3)
    plotXYResolutionOnResolution(regex_x, regex_y, labelX=labelX, labelY=labelY, labelXY=labelXY, unit=unit,
                                 part='electron', nbins=nbins, value=value,
                                 fit=fit, fitparam=fitparams[2], mark_x=mark_x, mark_y=mark_y)
    fig.tight_layout()

# ============ TOOLS ================ #


def _printProgressBar(iteration, total, prefix='', suffix='', decimals=1, length=100, fill='█', printEnd="\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = printEnd)
    # Print New Line on Complete
    if iteration == total:
        print()