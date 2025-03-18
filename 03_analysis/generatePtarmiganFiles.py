import numpy as _np
import matplotlib.pyplot as _plt
import h5py as _h5
import jinja2 as _jj
import glob as _glob
import pickle as _pk
import os

yml_default_path = "../../00_ptarmigan/pickles/"
template_default_path = "../../00_ptarmigan/"
template_default_file_name = "luxe_template.yml"

# ============ GENERATE FILES ================ #


def GeneratePtarmiganFile(tag="luxe_default_jinja", seed=0, ngenerate=10000,
                          offset=[0.0, 0.0, 0.0], radius=5e-6, angle=-17.2, charge=250e-12, E=16.5e9, DE=16.5e6, length=24e-6,
                          yml_path=yml_default_path, templatefilename=template_default_file_name, templatefolder=template_default_path):
    ymlfilename = yml_path + tag + ".yml"
    paramdict = dict(ident=tag, rng_seed=seed, n=ngenerate, offset=offset, radius=radius, collision_angle=angle,
                     charge=charge, E=E, DE=DE, length=length)
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
            raise ValueError("Number of cores required '{}' is above 8".format(mpi_nb))
        command = "mpirun -n {} ptarmigan ".format(mpi_nb)
    else:
        command = "ptarmigan".format(mpi_nb)
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


def GenerateRunAnalyse(data_dict, tag='luxe_default_GenRunAn', seed=0, ngenerate=10000,
                       offset=[0.0, 0.0, 0.0], radius=5e-6, angle=-17.2, charge=250e-12, E=16.5e9, DE=16.5e6, length=24e-6,
                       yml_path=yml_default_path, templatefilename=template_default_file_name, templatefolder=template_default_path,
                       printPtarmigan=False, removeYmlFile=False, removeH5File=False, mpi=False, mpi_nb=8):
    ymlfilename = GeneratePtarmiganFile(tag=tag, seed=seed, ngenerate=ngenerate, offset=offset,
                                        radius=radius, angle=angle, charge=charge, E=E, DE=DE, length=length,
                                        yml_path=yml_path, templatefilename=templatefilename, templatefolder=templatefolder)
    RunPtarmiganFile(ymlfilename, printPtarmigan=printPtarmigan, removeYmlFile=removeYmlFile, mpi=mpi, mpi_nb=mpi_nb)

    AnalyseFile(ymlfilename, data_dict, removeH5File=removeH5File)


def ScanPositionJitter(tag="luxe_default_position_jitter_scan", ngenerate=10000, coord='X', mu=0, sigma=3e-6, npoints=100,
                       radius=5e-6, angle=-17.2, charge=250e-12, E=16.5e9, DE=16.5e6, length=24e-6,
                       yml_path=yml_default_path, templatefilename=template_default_file_name, templatefolder=template_default_path,
                       printPtarmigan=False, removeYmlFile=False, removeH5File=False, mpi=False, mpi_nb=8):
    data_dict = {}
    printtag = tag + '_' + coord + '_' + str(ngenerate)
    for seed in range(npoints):
        fulltag = str(seed).zfill(3) + '_' + tag + '_' + coord + '_' + str(ngenerate)
        value = _np.random.normal(mu, sigma)
        _printProgressBar(seed, npoints, prefix='Run position jitter scan : {}'.format(printtag), suffix='Complete', length=50)
        GenerateRunAnalyse(data_dict, tag=fulltag, seed=seed, ngenerate=ngenerate, offset=SetOffset(value, coord),
                           radius=radius, angle=angle, charge=charge, E=E, DE=DE, length=length,
                           yml_path=yml_path, templatefilename=templatefilename, templatefolder=templatefolder,
                           printPtarmigan=printPtarmigan, removeYmlFile=removeYmlFile, removeH5File=removeH5File, mpi=mpi, mpi_nb=mpi_nb)
    picklefilename = (yml_path + tag + '_' + coord + '_n_' + str(ngenerate) + '_npoints_' + str(npoints) + '_sigma_' + str(sigma))
    WritePicke(data_dict, picklefilename)
    _printProgressBar(npoints, npoints, prefix='Run position jitter scan : {}'.format(printtag), suffix='Complete', length=50)


def ScanAngleJitter(tag="luxe_default_angle_jitter_scan", ngenerate=10000, mu=-17.2, sigma=6e-5, npoints=100,
                    radius=5e-6, offset=[0.0, 0.0, 0.0], charge=250e-12, E=16.5e9, DE=16.5e6, length=24e-6,
                    yml_path=yml_default_path, templatefilename=template_default_file_name, templatefolder=template_default_path,
                    printPtarmigan=False, removeYmlFile=False, removeH5File=False, mpi=False, mpi_nb=8):
    data_dict = {}
    printtag = tag + '_' + str(ngenerate)
    for seed in range(npoints):
        fulltag = str(seed).zfill(3) + '_' + tag + '_' + str(ngenerate)
        angle = _np.random.normal(mu, sigma)
        _printProgressBar(seed, npoints, prefix='Run angle jitter scan : {}'.format(printtag), suffix='Complete', length=50)
        GenerateRunAnalyse(data_dict, tag=fulltag, seed=seed, ngenerate=ngenerate, offset=offset,
                           radius=radius, angle=angle, charge=charge, E=E, DE=DE, length=length,
                           yml_path=yml_path, templatefilename=templatefilename, templatefolder=templatefolder,
                           printPtarmigan=printPtarmigan, removeYmlFile=removeYmlFile, removeH5File=removeH5File, mpi=mpi, mpi_nb=mpi_nb)
    picklefilename = (yml_path + tag + '_n_' + str(ngenerate) + '_npoints_' + str(npoints) + '_sigma_' + str(sigma))
    WritePicke(data_dict, picklefilename)
    _printProgressBar(npoints, npoints, prefix='Run angle jitter scan : {}'.format(printtag), suffix='Complete', length=50)


def ScanEnergyJitter(tag="luxe_default_energy_jitter_scan", ngenerate=10000, mu=0, sigma=1e-6, npoints=100,
                     yml_path=yml_default_path, templatefilename=template_default_file_name, templatefolder=template_default_path,
                     printPtarmigan=False, removeYmlFile=False, removeH5File=False, mpi=False, mpi_nb=8):
    data_dict = {}
    printtag = tag + '_' + str(ngenerate)
    for seed in range(npoints):
        fulltag = str(seed).zfill(3) + '_' + tag + '_' + str(ngenerate)
        DE = _np.random.normal(mu, sigma)
        _printProgressBar(seed, npoints, prefix='Run energy jitter scan : {}'.format(printtag), suffix='Complete', length=50)
        GenerateRunAnalyse(data_dict, tag=fulltag, seed=seed, ngenerate=ngenerate, DE=DE,
                           yml_path=yml_path, templatefilename=templatefilename, templatefolder=templatefolder,
                           printPtarmigan=printPtarmigan, removeYmlFile=removeYmlFile, removeH5File=removeH5File, mpi=mpi, mpi_nb=mpi_nb)
    picklefilename = (yml_path + tag + '_n_' + str(ngenerate) + '_npoints_' + str(npoints) + '_sigma_' + str(sigma))
    WritePicke(data_dict, picklefilename)
    _printProgressBar(npoints, npoints, prefix='Run energy jitter scan : {}'.format(printtag), suffix='Complete', length=50)


def ScanChargeJitter(tag="luxe_default_charge_jitter_scan", ngenerate=10000, mu=250e-12, sigma=1e-12, npoints=100,
                     yml_path=yml_default_path, templatefilename=template_default_file_name, templatefolder=template_default_path,
                     printPtarmigan=False, removeYmlFile=False, removeH5File=False, mpi=False, mpi_nb=8):
    data_dict = {}
    printtag = tag + '_' + str(ngenerate)
    for seed in range(npoints):
        fulltag = str(seed).zfill(3) + '_' + tag + '_' + str(ngenerate)
        charge = _np.random.normal(mu, sigma)
        _printProgressBar(seed, npoints, prefix='Run charge jitter scan : {}'.format(printtag), suffix='Complete', length=50)
        GenerateRunAnalyse(data_dict, tag=fulltag, seed=seed, ngenerate=ngenerate, charge=charge,
                           yml_path=yml_path, templatefilename=templatefilename, templatefolder=templatefolder,
                           printPtarmigan=printPtarmigan, removeYmlFile=removeYmlFile, removeH5File=removeH5File, mpi=mpi, mpi_nb=mpi_nb)
    picklefilename = (yml_path + tag + '_n_' + str(ngenerate) + '_npoints_' + str(npoints) + '_sigma_' + str(sigma))
    WritePicke(data_dict, picklefilename)
    _printProgressBar(npoints, npoints, prefix='Run charge jitter scan : {}'.format(printtag), suffix='Complete', length=50)


def ScanLengthJitter(tag="luxe_default_length_jitter_scan", ngenerate=10000, mu=40e-15, sigma=3e-15, npoints=100,
                     yml_path=yml_default_path, templatefilename=template_default_file_name, templatefolder=template_default_path,
                     printPtarmigan=False, removeYmlFile=False, removeH5File=False, mpi=False, mpi_nb=8):
    data_dict = {}
    printtag = tag + '_' + str(ngenerate)
    for seed in range(npoints):
        fulltag = str(seed).zfill(3) + '_' + tag + '_' + str(ngenerate)
        length = _np.random.normal(mu, sigma)
        _printProgressBar(seed, npoints, prefix='Run length jitter scan : {}'.format(printtag), suffix='Complete', length=50)
        GenerateRunAnalyse(data_dict, tag=fulltag, seed=seed, ngenerate=ngenerate, length=length,
                           yml_path=yml_path, templatefilename=templatefilename, templatefolder=templatefolder,
                           printPtarmigan=printPtarmigan, removeYmlFile=removeYmlFile, removeH5File=removeH5File, mpi=mpi, mpi_nb=mpi_nb)
    picklefilename = (yml_path + tag + '_n_' + str(ngenerate) + '_npoints_' + str(npoints) + '_sigma_' + str(sigma))
    WritePicke(data_dict, picklefilename)
    _printProgressBar(npoints, npoints, prefix='Run length jitter scan : {}'.format(printtag), suffix='Complete', length=50)


def ScanSizeJitter(tag="luxe_default_size_jitter_scan", ngenerate=10000, mu=5e-6, sigma=1e-6, npoints=100,
                     yml_path=yml_default_path, templatefilename=template_default_file_name, templatefolder=template_default_path,
                     printPtarmigan=False, removeYmlFile=False, removeH5File=False, mpi=False, mpi_nb=8):
    data_dict = {}
    printtag = tag + '_' + str(ngenerate)
    for seed in range(npoints):
        fulltag = str(seed).zfill(3) + '_' + tag + '_' + str(ngenerate)
        size = _np.random.normal(mu, sigma)
        _printProgressBar(seed, npoints, prefix='Run size jitter scan : {}'.format(printtag), suffix='Complete', length=50)
        GenerateRunAnalyse(data_dict, tag=fulltag, seed=seed, ngenerate=ngenerate, radius=size,
                           yml_path=yml_path, templatefilename=templatefilename, templatefolder=templatefolder,
                           printPtarmigan=printPtarmigan, removeYmlFile=removeYmlFile, removeH5File=removeH5File, mpi=mpi, mpi_nb=mpi_nb)
    picklefilename = (yml_path + tag + '_n_' + str(ngenerate) + '_npoints_' + str(npoints) + '_sigma_' + str(sigma))
    WritePicke(data_dict, picklefilename)
    _printProgressBar(npoints, npoints, prefix='Run size jitter scan : {}'.format(printtag), suffix='Complete', length=50)


def ScanPositionOffset(tag="luxe_default_position_offset_scan", seed=0, ngenerate=10000, coord='X', offset_list=[],
                       radius=5e-6, angle=-17.2,
                       yml_path=yml_default_path, templatefilename=template_default_file_name, templatefolder=template_default_path,
                       printPtarmigan=False, removeYmlFile=False, removeH5File=False, mpi=False, mpi_nb=8):
    data_dict = {}
    noffset = len(offset_list)
    printtag = tag + '_' + coord + '_' + str(ngenerate)
    for i, offset in enumerate(offset_list):
        fulltag = str(seed).zfill(3) + '_' + tag + '_' + coord + '_' + str(offset) + '_' + str(ngenerate)
        _printProgressBar(i, noffset, prefix='Run position offset scan : {}'.format(printtag), suffix='Complete', length=50)
        GenerateRunAnalyse(data_dict, tag=fulltag, seed=seed, ngenerate=ngenerate, offset=SetOffset(offset, coord), radius=radius, angle=angle,
                           yml_path=yml_path, templatefilename=templatefilename, templatefolder=templatefolder,
                           printPtarmigan=printPtarmigan, removeYmlFile=removeYmlFile, removeH5File=removeH5File, mpi=mpi, mpi_nb=mpi_nb)
        picklefilename = (yml_path + str(seed).zfill(3) + '_' + tag + '_' + coord + '_n_' + str(ngenerate))
        WritePicke(data_dict, picklefilename)
    _printProgressBar(noffset, noffset, prefix='Run position offset scan : {}'.format(printtag), suffix='Complete', length=50)


def ScanRadiusOffset(tag="luxe_default_radius_offset_scan", seed=0, ngenerate=10000, radius_list=[], offset=[0.0, 0.0, 0.0],
                     yml_path=yml_default_path, templatefilename=template_default_file_name, templatefolder=template_default_path,
                     printPtarmigan=False, removeYmlFile=False, removeH5File=False, mpi=False, mpi_nb=8):
    data_dict = {}
    nradius = len(radius_list)
    printtag = tag + '_' + str(ngenerate)
    for i, radius in enumerate(radius_list):
        fulltag = str(seed).zfill(3) + '_' + tag + '_' + str(radius) + '_' + str(ngenerate)
        _printProgressBar(i, nradius, prefix='Run radius offset scan : {}'.format(printtag), suffix='Complete', length=50)
        GenerateRunAnalyse(data_dict, tag=fulltag, seed=seed, ngenerate=ngenerate, offset=offset, radius=radius,
                           yml_path=yml_path, templatefilename=templatefilename, templatefolder=templatefolder,
                           printPtarmigan=printPtarmigan, removeYmlFile=removeYmlFile, removeH5File=removeH5File, mpi=mpi, mpi_nb=mpi_nb)
        picklefilename = (yml_path + str(seed).zfill(3) + '_' + tag + '_n_' + str(ngenerate))
        WritePicke(data_dict, picklefilename)
    _printProgressBar(nradius, nradius, prefix='Run radius offset scan : {}'.format(printtag), suffix='Complete', length=50)

# ============ ANALYSE FILES ================ #


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


def getNumberParticlesWeight(file, part=None):
    weight = getOutput(file, part=part, param='weight')
    return weight.sum()


def fillDataDict(data_dict, file):
    offsets, offset_unit = getAllBeamOffset(file, returnUnit=True)
    radius, radius_unit = getBeamRadius(file, returnUnit=True)
    angle, angle_unit = getBeamAngle(file, returnUnit=True)
    charge, charge_unit = getBeamCharge(file, returnUnit=True)
    length, length_unit = getBeamLength(file, returnUnit=True)
    data_dict["offset_unit"] = offset_unit
    data_dict["radius_unit"] = radius_unit
    data_dict["angle_unit"]  = angle_unit
    data_dict["charge_unit"] = charge_unit
    data_dict["length_unit"] = length_unit
    try:
        data_dict["OffsetX"].append(offsets[0])
        data_dict["OffsetY"].append(offsets[1])
        data_dict["OffsetZ"].append(offsets[2])
        data_dict["Radius"].append(radius)
        data_dict["Angle"].append(angle)
        data_dict["Charge"].append(charge)
        data_dict["Length"].append(length)
        data_dict["Nbelectron"].append(getNumberParticlesWeight(file, part='electron'))
        data_dict["Nbphoton"].append(getNumberParticlesWeight(file, part='photon'))
        data_dict["Nbpositron"].append(getNumberParticlesWeight(file, part='positron'))
    except KeyError:
        data_dict["OffsetX"]    = [offsets[0]]
        data_dict["OffsetY"]    = [offsets[1]]
        data_dict["OffsetZ"]    = [offsets[2]]
        data_dict["Radius"]     = [radius]
        data_dict["Angle"]      = [angle]
        data_dict["Charge"]     = [charge]
        data_dict["Length"]     = [length]
        data_dict["Nbelectron"] = [getNumberParticlesWeight(file, part='electron')]
        data_dict["Nbphoton"]   = [getNumberParticlesWeight(file, part='photon')]
        data_dict["Nbpositron"] = [getNumberParticlesWeight(file, part='positron')]


def AnalyseFile(ymlfilename, data_dict, removeH5File=False):
    h5filename = _glob.glob('{}*'.format(".." + ymlfilename.strip(".yml")))[0]
    fillDataDict(data_dict, h5filename)
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

# ============ WRITE/READ ================ #


def WritePicke(data_dict, outputfilename):
    with open('{}.pickle'.format(outputfilename), 'wb') as file:
        _pk.dump(data_dict, file, protocol=_pk.HIGHEST_PROTOCOL)


def ReadPicke(inputfilename):
    with open('{}'.format(inputfilename), 'rb') as file:
        data_dict = _pk.load(file)
    return data_dict

# ============ PLOTS ================ #


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


def plotCumulativeLength(inputfilename, X='Nbphoton', nbins=30, value=0.68, cumulative=True):
    data_dict = ReadPicke(inputfilename)
    upper_cl, lower_cl, mode = calcCumulativeLength(data_dict[X])
    variation = calcVariation(lower_cl, upper_cl, mode)

    _plt.hist(data_dict[X], bins=nbins, histtype='step', label="Signal {}".format(X))
    if cumulative:
        _plt.hist(data_dict[X], bins=nbins, histtype='step', color='k', alpha=0.3, cumulative=True, label="Cumulative")

    _plt.xlabel(X)
    _plt.axvline(upper_cl, ls='--', color='k', alpha=0.3, label="CL at {} : {:.2f} %".format(value, variation))
    _plt.axvline(lower_cl, ls='--', color='k', alpha=0.3)
    _plt.legend()


def plotAllParticlesCL(inputfilename, nbins=30, value=0.68, cumulative=True, figsize=[8, 5]):
    _plt.rcParams['font.size'] = 15
    _plt.figure(figsize=figsize)
    _plt.subplot(1, 3, 1)
    plotCumulativeLength(inputfilename, X='Nbphoton', nbins=nbins, value=value, cumulative=cumulative)
    _plt.subplot(1, 3, 2)
    plotCumulativeLength(inputfilename, X='Nbpositron', nbins=nbins, value=value, cumulative=cumulative)
    _plt.subplot(1, 3, 3)
    plotCumulativeLength(inputfilename, X='Nbelectron', nbins=nbins, value=value, cumulative=cumulative)

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