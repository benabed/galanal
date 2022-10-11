import galanal as gnl 
import numpy as nm
import optimize as opt
import pmclite as pmc


rq = gnl.rq("data/rq_spt_winter.cldf/")
lrange = gnl.lrange_class(100,1500,delta=30)
cl = "attic/base_plikHM_TTTEEE_lowl_lowE_lensing.minimum.theory_cl"
parbase_T = ["Ddust","beta_d","alpha","T","CIB","PS","CAL"]

# make problem, using fully empirical cov
pb143_217_353_545 = gnl.make_problem(rq,"TT",gnl.get_fqc([143,217,353,545],[353,545],outer=True,exclude=([[143,353]])),lrange,cl,parbase_T)
lkl143_217_353_545LRG = pmc.add_lkl(pmc.repar_lkl(pb143_217_353_545[0],{"cal_143":1,"T":19.6}),("gauss",[-2.55,1.48,1.,1.,1.],[.1**2,.02**2,.02**2,.02**2,.01**2],["alpha","beta_d","cal_545","cal_353","cal_217"]))

init_guess = {'Ddust': 1.7407949100938531,
 'alpha': -2.5027631088322884,
 'Acib_217_353': 432.9875079975021,
 'Acib_217_545': 4497.379472497491,
 'Acib_353_353': 2543.4217276089653,
 'Acib_353_545': 30818.399897530027,
 'Acib_545_545': 403311.7198208173,
 'Aps_217_353': 4.091338876694438,
 'Aps_217_545': 3824.028404713797,
 'Aps_353_353': 4915.540562088363,
 'Aps_353_545': 67033.9820382427,
 'Aps_545_545': 911476.3859404493,
 'T': 19.4,
 'beta_d': 1.48,
 'Acib_217_217': -26.510127297815583,
 'Aps_217_217': 234.6201395150361,
 'cal_217': 1.0014289141065136,
 'cal_353': 0.9989897430476743,
 'Acib_143_545': 500,
 'Aps_143_545': 300,
 'cal_545': 1}

init_cov = nm.array([[ 5.36795946e-03, -1.56241592e-05, -1.78124071e-03,
         3.60558290e+00, -1.43156340e-01,  2.61259945e+00,
         6.82400312e-01, -1.18302388e+01, -3.11676729e+02,
        -1.14580559e+00,  1.14316908e+00,  1.79435348e+00,
        -1.02043036e+00,  1.15093269e+02,  1.78165480e+03,
         3.70934651e-05, -3.74033620e-05,  3.90980598e-05],
       [-1.56241592e-05,  2.05148866e-07,  7.90849335e-06,
        -3.55084629e-02,  5.14030393e-03, -1.60184956e-02,
         1.98351141e-03,  2.23477251e-02,  8.34735252e-01,
         1.26718553e-02, -1.99451132e-02, -5.29764050e-02,
        -9.44696839e-03, -2.90333421e-01, -4.65649155e+00,
        -6.56148150e-07,  5.11463568e-08, -2.69885529e-07],
       [-1.78124071e-03,  7.90849335e-06,  7.16650769e-04,
        -1.44745505e+00,  3.73252626e-01, -4.51677759e-01,
         1.67819655e-01,  2.72457927e+00,  9.62335509e+01,
         5.73638164e-01, -1.61007731e+00, -3.80876752e+00,
        -8.95891779e-01, -3.37631887e+01, -5.38627170e+02,
        -2.77605075e-05,  6.78454495e-06, -3.56851297e-05],
       [ 3.60558290e+00, -3.55084629e-02, -1.44745505e+00,
         1.50425341e+04,  1.24055154e+03,  1.36983452e+04,
        -6.30747424e+02, -4.65384332e+03, -1.89396551e+05,
        -6.49323183e+02, -3.58129728e+03, -7.54944094e+03,
         2.70391249e+03,  6.49022302e+04,  1.05232951e+06,
         5.62824790e-02, -9.43592575e-03, -1.07185310e-01],
       [-1.43156340e-01,  5.14030393e-03,  3.73252626e-01,
         1.24055154e+03,  1.83100219e+03,  4.90572343e+03,
         5.43269883e+02, -1.40452402e+03, -1.89657600e+03,
         8.91077108e+02, -5.57789774e+03, -1.38639415e+04,
        -1.73818829e+03,  3.17589667e+03,  2.76970954e+04,
        -1.41436003e-01, -7.01648266e-03, -9.03211236e-02],
       [ 2.61259945e+00, -1.60184956e-02, -4.51677759e-01,
         1.36983452e+04,  4.90572343e+03,  2.24462929e+04,
        -9.88978883e+02, -1.78466236e+03, -1.27434231e+05,
         1.08157099e+03, -1.50388122e+04, -3.53351337e+04,
         3.60053434e+03,  4.08460649e+04,  6.92714241e+05,
        -3.19077596e-01,  5.31054853e-04, -2.28722297e-01],
       [ 6.82400312e-01,  1.98351141e-03,  1.67819655e-01,
        -6.30747424e+02,  5.43269883e+02, -9.88978883e+02,
         6.87013801e+03, -2.21266137e+04, -1.63433815e+05,
         1.08918657e+01, -1.34623210e+03, -1.52522699e+03,
        -2.13423928e+04,  9.10736022e+04,  1.12960640e+06,
        -2.25390178e-02, -1.00538427e-01, -1.42347924e-01],
       [-1.18302388e+01,  2.23477251e-02,  2.72457927e+00,
        -4.65384332e+03, -1.40452402e+03, -1.78466236e+03,
        -2.21266137e+04,  8.87689679e+04,  1.06330231e+06,
         2.04413739e+03,  2.06110370e+03,  1.48962847e+03,
         6.67392403e+04, -4.86047285e+05, -6.67215052e+06,
         1.84079939e-03,  3.73631519e-01,  3.62326780e-01],
       [-3.11676729e+02,  8.34735252e-01,  9.62335509e+01,
        -1.89396551e+05, -1.89657600e+03, -1.27434231e+05,
        -1.63433815e+05,  1.06330231e+06,  2.03565529e+07,
         6.36849725e+04, -3.92887511e+04, -7.25134513e+04,
         4.46427060e+05, -8.07784114e+06, -1.19931032e+08,
        -1.65844380e+00,  3.91999178e+00,  4.16513340e-01],
       [-1.14580559e+00,  1.26718553e-02,  5.73638164e-01,
        -6.49323183e+02,  8.91077108e+02,  1.08157099e+03,
         1.08918657e+01,  2.04413739e+03,  6.36849725e+04,
         1.43443422e+03, -2.54770513e+03, -6.84556676e+03,
        -2.75128929e+02, -2.28334467e+04, -3.59262331e+05,
        -6.71738592e-02,  5.70678556e-03, -5.02770211e-02],
       [ 1.14316908e+00, -1.99451132e-02, -1.61007731e+00,
        -3.58129728e+03, -5.57789774e+03, -1.50388122e+04,
        -1.34623210e+03,  2.06110370e+03, -3.92887511e+04,
        -2.54770513e+03,  1.96748969e+04,  4.87146861e+04,
         4.48047815e+03,  7.99216927e+03,  1.79250497e+05,
         3.53558156e-01,  1.35572200e-02,  2.83231267e-01],
       [ 1.79435348e+00, -5.29764050e-02, -3.80876752e+00,
        -7.54944094e+03, -1.38639415e+04, -3.53351337e+04,
        -1.52522699e+03,  1.48962847e+03, -7.25134513e+04,
        -6.84556676e+03,  4.87146861e+04,  1.25338534e+05,
         5.22349558e+03,  1.94287284e+04,  3.58129832e+05,
         7.80036313e-01,  1.36871832e-02,  6.63047304e-01],
       [-1.02043036e+00, -9.44696839e-03, -8.95891779e-01,
         2.70391249e+03, -1.73818829e+03,  3.60053434e+03,
        -2.13423928e+04,  6.67392403e+04,  4.46427060e+05,
        -2.75128929e+02,  4.48047815e+03,  5.22349558e+03,
         6.65292461e+04, -2.60925121e+05, -3.16286225e+06,
         7.87092745e-02,  3.06640170e-01,  4.53724981e-01],
       [ 1.15093269e+02, -2.90333421e-01, -3.37631887e+01,
         6.49022302e+04,  3.17589667e+03,  4.08460649e+04,
         9.10736022e+04, -4.86047285e+05, -8.07784114e+06,
        -2.28334467e+04,  7.99216927e+03,  1.94287284e+04,
        -2.60925121e+05,  3.32912890e+06,  4.83772995e+07,
         4.92472260e-01, -1.88134284e+00, -8.17671096e-01],
       [ 1.78165480e+03, -4.65649155e+00, -5.38627170e+02,
         1.05232951e+06,  2.76970954e+04,  6.92714241e+05,
         1.12960640e+06, -6.67215052e+06, -1.19931032e+08,
        -3.59262331e+05,  1.79250497e+05,  3.58129832e+05,
        -3.16286225e+06,  4.83772995e+07,  7.11578123e+08,
         8.65688378e+00, -2.51666580e+01, -6.65786598e+00],
       [ 3.70934651e-05, -6.56148150e-07, -2.77605075e-05,
         5.62824790e-02, -1.41436003e-01, -3.19077596e-01,
        -2.25390178e-02,  1.84079939e-03, -1.65844380e+00,
        -6.71738592e-02,  3.53558156e-01,  7.80036313e-01,
         7.87092745e-02,  4.92472260e-01,  8.65688378e+00,
         2.21097467e-05,  1.21669240e-07,  4.14611855e-06],
       [-3.74033620e-05,  5.11463568e-08,  6.78454495e-06,
        -9.43592575e-03, -7.01648266e-03,  5.31054853e-04,
        -1.00538427e-01,  3.73631519e-01,  3.91999178e+00,
         5.70678556e-03,  1.35572200e-02,  1.36871832e-02,
         3.06640170e-01, -1.88134284e+00, -2.51666580e+01,
         1.21669240e-07,  1.61317329e-06,  1.81199646e-06],
       [ 3.90980598e-05, -2.69885529e-07, -3.56851297e-05,
        -1.07185310e-01, -9.03211236e-02, -2.28722297e-01,
        -1.42347924e-01,  3.62326780e-01,  4.16513340e-01,
        -5.02770211e-02,  2.83231267e-01,  6.63047304e-01,
         4.53724981e-01, -8.17671096e-01, -6.65786598e+00,
         4.14611855e-06,  1.81199646e-06,  1.12103590e-05]])

# first chain, used to compute improved covariance
ch143_217_353_545LRG =  pmc.mcmc(lkl143_217_353_545LRG,init_guess,init_cov,nstep=5000)

# improve covariance
gnl.create_fake_data(rq,"TT",gnl.get_fqc([143,217,353,545],[353,545],outer=True,exclude=([[143,353]])),lrange,100,cl,parbase_T,ch143_217_353_545LRG.meandict()|{"cal_143":1,"T":19.6})

# build new covariance with first guess model
pb143_217_353_545F = gnl.make_problem(rq,"TT",gnl.get_fqc([143,217,353,545],[353,545],outer=True,exclude=([[143,353]])),lrange,cl,parbase_T,fake=True,regularize_var=True)
lkl143_217_353_545LRGF = pmc.add_lkl(pmc.repar_lkl(pb143_217_353_545F[0],{"cal_143":1,"T":19.6}),("gauss",[-2.55,1.48,1.,1.,1.],[.1**2,.02**2,.02**2,.02**2,.01**2],["alpha","beta_d","cal_545","cal_353","cal_217"]))

ch143_217_353_545LRGF =  pmc.mcmc(lkl143_217_353_545LRGF,init_guess,init_cov,nstep=5000)

rq2 = gnl.rq("data/rq_spt_winter.cldf/")
gnl.create_fake_data(rq2,"TT",gnl.get_fqc([143,217,353,545],[353,545],outer=True,exclude=([[143,353]])),lrange,100,cl,parbase_T,ch143_217_353_545LRGF.meandict()|{"cal_143":1,"T":19.6})
pb143_217_353_545F2 = gnl.make_problem(rq2,"TT",gnl.get_fqc([143,217,353,545],[353,545],outer=True,exclude=([[143,353]])),lrange,cl,parbase_T,fake=True,regularize_var=True)
lkl143_217_353_545LRGF2 = pmc.add_lkl(pmc.repar_lkl(pb143_217_353_545F2[0],{"cal_143":1,"T":19.6}),("gauss",[-2.55,1.48,1.,1.,1.],[.1**2,.02**2,.02**2,.02**2,.01**2],["alpha","beta_d","cal_545","cal_353","cal_217"]))

ch143_217_353_545LRGF2 =  pmc.mcmc(lkl143_217_353_545LRGF2,init_guess,init_cov,nstep=5000)
ch143_217_353_545LRGF2_ =  pmc.mcmc(lkl143_217_353_545LRGF2,ch143_217_353_545LRGF2.covariance(),init_cov,nstep=20000)
ch143_217_353_545LRGF2_B =  pmc.mcmc(lkl143_217_353_545LRGF2,ch143_217_353_545LRGF2.covariance(),init_cov,nstep=20000)

ch143_217_353_545LRGF2_40K = pmc.join_chain(ch143_217_353_545LRGF2_,ch143_217_353_545LRGF2_B)

ch143_217_353_545LRGF2_40K.to_file("ch143_217_353_545LRGF2_40K.npz")

ch143_217_353_545LRGF2_40K = pmc.chain("ch143_217_353_545LRGF2_40K.npz")

res = ch143_217_353_545LRGF2_40K.meandict()
#res = {'Ddust': 1.7399270072427102,
# 'beta_d': 1.487055687227529,
# 'alpha': -2.499082847488835,
# 'Acib_143_545': 362.6137426636014,
# 'Acib_217_353': 438.3053968387038,
# 'Acib_217_545': 3196.239483827595,
# 'Acib_353_353': 2617.0885027876343,
# 'Acib_353_545': 30727.421026570286,
# 'Acib_545_545': 400724.5754726033,
# 'Aps_143_545': 3108.694546791206,
# 'Aps_217_353': 298.6922776559353,
# 'Aps_217_545': 10706.252781070973,
# 'Aps_353_353': 4807.274451968251,
# 'Aps_353_545': 68687.6369664545,
# 'Aps_545_545': 920997.3214875902,
# 'cal_217': 1.0073016570818527,
# 'cal_353': 0.9903810231704135,
# 'cal_545': 1.0024541344424698}

res_std = ch143_217_353_545LRGF2_40K.stddict()
#res_std = {'Ddust': 0.12832884981030332,
# 'beta_d': 0.01633690843366671,
# 'alpha': 0.044572609270231794,
# 'Acib_143_545': 406.7057485980303,
# 'Acib_217_353': 81.65186199358597,
# 'Acib_217_545': 428.65250729066116,
# 'Acib_353_353': 134.3599894017914,
# 'Acib_353_545': 494.81684717767683,
# 'Acib_545_545': 7983.595713287778,
# 'Aps_143_545': 1507.4798391583067,
# 'Aps_217_353': 257.185832174454,
# 'Aps_217_545': 1533.3861029483664,
# 'Aps_353_353': 353.8896230893053,
# 'Aps_353_545': 3415.336391074614,
# 'Aps_545_545': 48137.286973044625,
# 'cal_217': 0.007879255148913395,
# 'cal_353': 0.00828297693910585,
# 'cal_545': 0.011702610385403545}

lrangeB80 = gnl.lrange_class(100,1500,delta=80)
lrangeB50 = gnl.lrange_class(100,1500,delta=50)
pb143_217_353_545F2B50 = gnl.make_problem(rq2,"TT",gnl.get_fqc([143,217,353,545],[353,545],outer=True,exclude=([[143,353]])),lrangeB50,cl,parbase_T,fake=True,regularize_var=True)
pb143_217_353_545F2B80 = gnl.make_problem(rq2,"TT",gnl.get_fqc([143,217,353,545],[353,545],outer=True,exclude=([[143,353]])),lrangeB80,cl,parbase_T,fake=True,regularize_var=True)
lkl143_217_353_545LRGF2B80 = pmc.add_lkl(pmc.repar_lkl(pb143_217_353_545F2B80[0],{"cal_143":1,"T":19.6}),("gauss",[-2.55,1.48,1.,1.,1.],[.1**2,.02**2,.02**2,.02**2,.01**2],["alpha","beta_d","cal_545","cal_353","cal_217"]))
lkl143_217_353_545LRGF2B50 = pmc.add_lkl(pmc.repar_lkl(pb143_217_353_545F2B50[0],{"cal_143":1,"T":19.6}),("gauss",[-2.55,1.48,1.,1.,1.],[.1**2,.02**2,.02**2,.02**2,.01**2],["alpha","beta_d","cal_545","cal_353","cal_217"]))
ch143_217_353_545LRGF2B80 =  pmc.mcmc(lkl143_217_353_545LRGF2B80,ch143_217_353_545LRGF2_40K.meandict(),ch143_217_353_545LRGF2_40K.covariance(),nstep=40000)
ch143_217_353_545LRGF2B50 =  pmc.mcmc(lkl143_217_353_545LRGF2B50,ch143_217_353_545LRGF2_40K.meandict(),ch143_217_353_545LRGF2_40K.covariance(),nstep=40000)



from matplotlib import rcParams, rc

# common setup for matplotlib
params = {#'backend': 'pdf',
          'savefig.dpi': 300, # save figures to 300 dpi
          'axes.labelsize': 10,
          'font.size': 10,
          'legend.fontsize': 8,
          'xtick.labelsize': 10,
          'ytick.major.pad': 6,
          'xtick.major.pad': 6,
          'ytick.labelsize': 10,
          'text.usetex': True,
          'font.family':'sans-serif',
          # free font similar to Helvetica
          'font.sans-serif':'FreeSans'}

# use of Sans Serif also in math mode
rc('text.latex', preamble=r'\usepackage{sfmath}')

rcParams.update(params)

import matplotlib.pyplot as plt

def cm2inch(cm):
    """Centimeters to inches"""
    return cm *0.393701

plt.figure(figsize=(cm2inch(10)*1.5,cm2inch(10)*1.5*2))
gnl.plotpb(pb143_217_353_545F2B50,"TT",gnl.get_fqc([143,217,353,545],[353,545],outer=True,exclude=([[143,353]])),lrangeB50,ch143_217_353_545LRGF2B50.meandict()|{"cal_143":1,"T":19.6},fig=-1)
plt.ylim(.4e3,7e5)
plt.ylabel("$\\mathcal{D}_\ell\ [\mu K^2]$")
plt.xlabel("$\ell$")
plt.title("$D^{\mathrm{dust}}_{80} = %.3g \pm %.2g, \\alpha =  %.3g \pm %.2g, \\beta = %.3g \pm %.2g$"%(ch143_217_353_545LRGF2B50.meandict()["Ddust"],ch143_217_353_545LRGF2B50.stddict()["Ddust"],ch143_217_353_545LRGF2B50.meandict()["alpha"],ch143_217_353_545LRGF2B50.stddict()["alpha"],ch143_217_353_545LRGF2B50.meandict()["beta_d"],ch143_217_353_545LRGF2B50.stddict()["beta_d"]))
plt.xticks((100,300,1000,1500),("100","300","$1000$","1500"))
plt.xlim(100,1700)
plt.tight_layout()
plt.savefig("BF_143_217_353_545LRG_clB50.pdf")

nocib = {
  'Acib_143_545': 0,
  'Acib_217_353': 0,
  'Acib_217_545': 0,
  'Acib_353_353': 0,
  'Acib_353_545': 0,
  'Acib_545_545': 0,
  'beta_d': 1.487055687227529,
  'alpha': -2.499082847488835,
  'cal_217': 1.0073016570818527,
  'cal_353': 0.9903810231704135,
  'cal_545': 1.0024541344424698}
lkl143_217_353_545NOCIBF2 = (pmc.repar_lkl(pb143_217_353_545F2[0],{"cal_143":1,"T":19.6}|nocib))
lkl143_217_353_545NOCIBF2B50 = (pmc.repar_lkl(pb143_217_353_545F2B50[0],{"cal_143":1,"T":19.6}|nocib))
ch143_217_353_545NOCIBF2B50 =  pmc.mcmc(lkl143_217_353_545NOCIBF2B50,ch143_217_353_545NOCIBF2R.meandict(),ch143_217_353_545NOCIBF2R.covariance(),nstep=20000)
ch143_217_353_545NOCIBF2B50.to_file("ch143_217_353_545LRG_NOCIB_B50.npz")

covNOCIB =nm. array([[ 1.80470183e-03, -7.68999834e-01, -1.23209164e-01,
        -4.09632882e+00, -2.43501417e+00, -2.64171610e+01,
        -5.79394504e+02],
       [-7.68999834e-01,  4.05554216e+05,  2.44307831e+04,
         3.88416565e+05,  3.79781101e+04,  4.83191173e+05,
         1.53258741e+06],
       [-1.23209164e-01,  2.44307831e+04,  4.09723742e+03,
         2.69869890e+04,  8.14960282e+03,  4.69942113e+04,
         3.26959249e+05],
       [-4.09632882e+00,  3.88416565e+05,  2.69869890e+04,
         4.23160815e+05,  5.43992405e+04,  6.72743885e+05,
         4.74176421e+06],
       [-2.43501417e+00,  3.79781101e+04,  8.14960282e+03,
         5.43992405e+04,  4.75623157e+04,  1.34801140e+05,
         2.17978124e+06],
       [-2.64171610e+01,  4.83191173e+05,  4.69942113e+04,
         6.72743885e+05,  1.34801140e+05,  2.92283712e+06,
         3.25478348e+07],
       [-5.79394504e+02,  1.53258741e+06,  3.26959249e+05,
         4.74176421e+06,  2.17978124e+06,  3.25478348e+07,
         5.16685059e+08]])

guess_nocib ={'Ddust': 2.9814499657822138,
 'Aps_143_545': 3657.393641549557,
 'Aps_217_353': 2128.294011590635,
 'Aps_217_545': 21462.74118842117,
 'Aps_353_353': 15158.693699631267,
 'Aps_353_545': 182443.158172296,
 'Aps_545_545': 2219489.1106524207}

ch143_217_353_545NOCIBF20K=  pmc.mcmc(lkl143_217_353_545NOCIBF2,guess_nocib,covNOCIB,nstep=20000)
ch143_217_353_545NOCIBF20K.to_file("ch143_217_353_545LRG_NOCIB_20K.npz")

ch143_217_353_545NOCIBF20K = pmc.chain("ch143_217_353_545LRG_NOCIB_20K.npz")
ch143_217_353_545NOCIBF2B50 =  pmc.mcmc(lkl143_217_353_545NOCIBF2B50,ch143_217_353_545NOCIBF2R.meandict(),ch143_217_353_545NOCIBF2R.covariance(),nstep=20000)

plt.figure(figsize=(cm2inch(10)*1.5,cm2inch(10)*1.5*3./4.))
pb = pb143_217_353_545F2B50
pars = ch143_217_353_545LRGF2B50.meandict()|{"cal_143":1,"T":19.6}
pars_nocib = ch143_217_353_545NOCIBF2B50.meandict()|{"cal_143":1,"T":19.6}|nocib
clrs = plt.get_cmap("tab20").colors

i=-1
label  ="$%d%s\\times%d%s$"%(545,"T",545,"T")
aa = plt.errorbar(lrangeB50.lm,pb[1][i]*lrangeB50.llp1,pb[2][0][i]*lrangeB50.llp1,label=label,marker="o",lw=.5,ls="",c=clrs[1])
tot545, = plt.loglog(lrangeB50.lm,pb[3][0](pars)[i]*lrangeB50.llp1,lw=2,c=clrs[0],label="total")
totnocib545, = plt.loglog(lrangeB50.lm,pb[3][0](pars_nocib)[i]*lrangeB50.llp1,lw=2,c=clrs[5],label="total no cib")
dst, = plt.loglog(lrangeB50.lm,pb[3][2](pars)[i]*lrangeB50.llp1,lw=2,ls="--",c=clrs[0],label="dust")
fg, = plt.loglog(lrangeB50.lm,pb[3][3](pars)[i]*lrangeB50.llp1,lw=1,ls="-.",c=clrs[0],label = "diffuse fg")
dstnocib, = plt.loglog(lrangeB50.lm,pb[3][2](pars_nocib)[i]*lrangeB50.llp1,lw=2,ls="--",c=clrs[5],label="dust no cib")
fgnocib, = plt.loglog(lrangeB50.lm,pb[3][3](pars_nocib)[i]*lrangeB50.llp1,lw=1,ls="-.",c=clrs[5],label = "diffuse fg no cib")

i=-2
label  ="$%d%s\\times%d%s$"%(353,"T",545,"T")
bb = plt.errorbar(lrangeB50.lm,pb[1][i]*lrangeB50.llp1,pb[2][0][i]*lrangeB50.llp1,label=label,marker="o",lw=.5,ls="",c=clrs[7])
tot353, = plt.loglog(lrangeB50.lm,pb[3][0](pars)[i]*lrangeB50.llp1,lw=2,c=clrs[6],label="total")
totnocib353, = plt.loglog(lrangeB50.lm,pb[3][0](pars_nocib)[i]*lrangeB50.llp1,lw=2,c=clrs[2],label="total no cib")
dst, = plt.loglog(lrangeB50.lm,pb[3][2](pars)[i]*lrangeB50.llp1,lw=2,ls="--",c=clrs[6],label="dust")
fg, = plt.loglog(lrangeB50.lm,pb[3][3](pars)[i]*lrangeB50.llp1,lw=1,ls="-.",c=clrs[6],label = "diffuse fg")
dstnocib, = plt.loglog(lrangeB50.lm,pb[3][2](pars_nocib)[i]*lrangeB50.llp1,lw=2,ls="--",c=clrs[2],label="dust no cib")
fgnocib, = plt.loglog(lrangeB50.lm,pb[3][3](pars_nocib)[i]*lrangeB50.llp1,lw=1,ls="-.",c=clrs[2],label = "diffuse fg no cib")
plt.legend(handles= [aa,bb,tot545,tot353,totnocib545,totnocib353,dst,fg],frameon=False,fontsize=8,ncol=4,loc=0)
ax = plt.gca()
leg = ax.get_legend()
for i in range(6,8):
  leg.legendHandles[i].set_color('grey') 
plt.ylim(5e3,8e5)
plt.ylabel("$\\mathcal{D}_\ell\ [\mu K^2]$")
plt.xlabel("$\ell$")
plt.title("$D^{\\mathrm{dust}}_{80} = %.3g \pm %.2g, D^{\\mathrm{dust, no cib}}_{80} = %.3g \pm %.2g$"%(ch143_217_353_545LRGF2B50.meandict()["Ddust"],ch143_217_353_545LRGF2B50.stddict()["Ddust"],ch143_217_353_545NOCIBF2B50.meandict()["Ddust"],ch143_217_353_545NOCIBF2B50.stddict()["Ddust"]))
plt.xticks((100,300,1000,1500),("100","300","$1000$","1500"))
plt.xlim(100,1700)
plt.tight_layout()
plt.savefig("NOCIB_143_217_353_545LRG_B50l.pdf")

plt.figure(figsize=(cm2inch(10)*1.5*2,cm2inch(10)*1.5))
plt.subplot(131)
ch143_217_353_545LRGF2B50.contour("Ddust","Acib_545_545",(1.35,2.45,40),(36e4,43e4),c="k")
plt.xlabel("$D_{80}^{\\mathrm{dust}}$")
plt.ylabel("$A_{\mathrm{CIB}}^{545\\times 545}$")

plt.subplot(132)
ch143_217_353_545LRGF2B50.contour("Ddust","Acib_353_545",(1.35,2.45,40),(29000,33000,40),c="k")
plt.xlabel("$D_{80}^{\\mathrm{dust}}$")
plt.ylabel("$A_{\mathrm{CIB}}^{353\\times 545}$")

plt.subplot(133)
ch143_217_353_545LRGF2B50.contour("Ddust","Acib_353_353",(1.35,2.45,40),(2000,3200,40),c="k")
plt.xlabel("$D_{80}^{\\mathrm{dust}}$")
plt.ylabel("$A_{\mathrm{CIB}}^{353\\times 353}$")
plt.tight_layout()
plt.savefig("DustXCIB.pdf")

pb100_143_217 = gnl.make_problem(rq2,"TT",gnl.get_fqc([100,143,217],[100,143,217],outer=True),lrange,cl,parbase_T,fake=False,regularize_var=True)
def_low = {'beta_d': 1.487055687227529,
    'alpha': -2.499082847488835,
    'cal_217': 1,
    'cal_143': 1,
    'cal_100': 1,
    "T":19.6,
    'Ddust': 1.7399270072427102}
  
lkl100_143_217 = (pmc.repar_lkl(pb100_143_217[0],def_low))
pos0 = [10,20,30,30,50,60,100,100,100,100,100,100]
cov0 = nm.diag(nm.array([10,20,30,30,50,60,100,100,100,100,100,100])/10.)**2
ch100_143_217 = pmc.mcmc(lkl100_143_217,pos0,cov0,nstep=2000)

lrangeB50 = gnl.lrange_class(100,1500,delta=50)
pb143_217_353_545F2B50 = gnl.make_problem(rq2,"TT",gnl.get_fqc([143,217,353,545],[353,545],outer=True,exclude=([[143,353]])),lrangeB50,cl,parbase_T,fake=True,regularize_var=True)
lkl143_217_353_545LRGF2B50 = pmc.add_lkl(pmc.repar_lkl(pb143_217_353_545F2B50[0],{"cal_143":1,"T":19.6}),("gauss",[-2.55,1.48,1.,1.,1.],[.1**2,.02**2,.02**2,.02**2,.01**2],["alpha","beta_d","cal_545","cal_353","cal_217"]))
ch143_217_353_545LRGF2B50 =  pmc.mcmc(lkl143_217_353_545LRGF2B50,ch143_217_353_545LRGF2_40K.meandict(),ch143_217_353_545LRGF2_40K.covariance(),nstep=20000)
