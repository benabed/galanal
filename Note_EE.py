import numpy as nm
import galanal as gnl
import pmclite as pmc

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


rq = gnl.rq("data/rq_spt_winter.cldf/")
lrange = gnl.lrange_class(100,1500,delta=30)
lrangeB50 = gnl.lrange_class(100,1500,delta=50)
lrangeB80 = gnl.lrange_class(100,1500,delta=80)
cl = "attic/base_plikHM_TTTEEE_lowl_lowE_lensing.minimum.theory_cl"

parbase_T = ["Ddust","beta_d","alpha","T","CAL"]
fqc = gnl.get_fqc([143,217,353],[217,353],outer=True,exclude=([[143,217]]))
pb143_217_353 = gnl.make_problem(rq,"EE",fqc,lrange,cl,parbase_T)

init_guess = {
  'Ddust'   : .05,        
  'beta_d'  : 1.53,         
  'alpha'   : -2.42,        
  'T'       : 19.6,    
  'cal_143' : 1,          
  'cal_217' : 1,          
  'cal_353' : 1}
init_cov = nm.array([.01,.02,.02,.02,.02])**2

lkl143_217_353 = pmc.add_lkl(pmc.repar_lkl(pb143_217_353[0],{"cal_143":1,"T":19.6}),("gauss",[-2.42,1.51,1.,1.],[.04**2,.04**2,.02**2,.02**2],["alpha","beta_d","cal_353","cal_217"]))
ch143_217_353 =  pmc.mcmc(lkl143_217_353,init_guess,init_cov,nstep=2000)
ch143_217_353 =  pmc.mcmc(lkl143_217_353,ch143_217_353.meandict(),ch143_217_353.covariance(),nstep=10000)
ch143_217_353.to_file("ch143_217_353_EE_pre.npz")

rq2 = gnl.rq("data/rq_spt_winter.cldf/")

gnl.create_fake_data(rq2,"EE",fqc,lrange,100,cl,parbase_T,ch143_217_353.meandict()|{"cal_143":1,"T":19.6})

pb143_217_3532 = gnl.make_problem(rq2,"EE",fqc,lrange,cl,parbase_T)
pb143_217_3532B50 = gnl.make_problem(rq2,"EE",fqc,lrangeB50,cl,parbase_T)
pb143_217_3532B80 = gnl.make_problem(rq2,"EE",fqc,lrangeB80,cl,parbase_T)

lkl143_217_3532 = pmc.add_lkl(pmc.repar_lkl(pb143_217_3532[0],{"cal_143":1,"T":19.6}),("gauss",[-2.42,1.51,1.,1.],[.04**2,.04**2,.02**2,.02**2],["alpha","beta_d","cal_353","cal_217"]))
lkl143_217_3532B50 = pmc.add_lkl(pmc.repar_lkl(pb143_217_3532B50[0],{"cal_143":1,"T":19.6}),("gauss",[-2.42,1.51,1.,1.],[.04**2,.04**2,.02**2,.02**2],["alpha","beta_d","cal_353","cal_217"]))
lkl143_217_3532B80 = pmc.add_lkl(pmc.repar_lkl(pb143_217_3532B80[0],{"cal_143":1,"T":19.6}),("gauss",[-2.42,1.51,1.,1.],[.04**2,.04**2,.02**2,.02**2],["alpha","beta_d","cal_353","cal_217"]))

ch143_217_3532 =  pmc.mcmc(lkl143_217_3532,ch143_217_353.meandict(),ch143_217_353.covariance(),nstep=40000)
ch143_217_3532.to_file("ch143_217_353_EE.npz")
ch143_217_3532B50 =  pmc.mcmc(lkl143_217_3532B50,ch143_217_3532.meandict(),ch143_217_3532.covariance(),nstep=40000)
ch143_217_3532.to_file("ch143_217_353_EE_B50.npz")
ch143_217_3532B80 =  pmc.mcmc(lkl143_217_3532B80,ch143_217_3532.meandict(),ch143_217_3532.covariance(),nstep=40000)
ch143_217_3532.to_file("ch143_217_353_EE_B80.npz")

ch143_217_3532.resdict()
##{'Ddust': (0.04692705092486124, 0.00720590599985159),
## 'beta_d': (1.5153300846694844, 0.03909597453003061),
## 'alpha': (-2.424150368082291, 0.03916158137829446),
## 'cal_217': (0.9716105730515276, 0.011283681321944848),
## 'cal_353': (1.013639249686314, 0.01825244551781138)}

ch143_217_3532B50.resdict()
##{'Ddust': (0.04976537848290014, 0.00754220918332044),
## 'beta_d': (1.5125826171435364, 0.04070893102896154),
## 'alpha': (-2.423236285309234, 0.04059701177299525),
## 'cal_217': (0.9721424279768744, 0.011193783229647583),
## 'cal_353': (1.012591022288744, 0.01881742508012086)}

ch143_217_3532B80.resdict()
##{'Ddust': (0.04978414433946677, 0.007451574802814308),
## 'beta_d': (1.5142105862324255, 0.03857219475675499),
## 'alpha': (-2.4250838134325323, 0.039513527059323546),
## 'cal_217': (0.9720258502262187, 0.011245726262035712),
## 'cal_353': (1.01446478079885, 0.018164989734582335)}

plt.figure(figsize=(cm2inch(10)*1.5,cm2inch(10)*1.5*1.5))
gnl.plotpb(pb143_217_3532B50,"EE",fqc,lrangeB50,ch143_217_3532B50.meandict()|{"cal_143":1,"T":19.6},fig=-1,loclegend=2)
plt.ylim(.5,200)
plt.ylabel("$\\mathcal{D}_\ell\ [\mu K^2]$")
plt.xlabel("$\ell$")
plt.title("$D^{\mathrm{dust}}_{80} = %.3g \pm %.2g, \\alpha =  %.3g \pm %.2g, \\beta = %.3g \pm %.2g$"%(ch143_217_3532B50.meandict()["Ddust"],ch143_217_3532B50.stddict()["Ddust"],ch143_217_3532B50.meandict()["alpha"],ch143_217_3532B50.stddict()["alpha"],ch143_217_3532B50.meandict()["beta_d"],ch143_217_3532B50.stddict()["beta_d"]))
plt.tight_layout()
plt.savefig("BF_143_217_353_EE_clB50.pdf")

lkl143_217_3532B50EXT = pmc.add_lkl(pmc.repar_lkl(pb143_217_3532B50[0],{"cal_143":1,"T":19.6}),("gauss",[-2.42,1.51,1.,1.],[.1**2,.1**2,.05**2,.05**2],["alpha","beta_d","cal_353","cal_217"]))
ch143_217_3532B50EXT =  pmc.mcmc(lkl143_217_3532B50EXT,ch143_217_3532B50.meandict(),ch143_217_3532B50.covariance(),nstep=40000)
ch143_217_3532B50EXT.to_file("ch143_217_353_EE_B50_EXT.npz")
ch143_217_3532B50EXT.resdict()
##{'Ddust': (0.047480278066954866, 0.00993535129099291),
## 'beta_d': (1.5207072974826967, 0.09472068880133235),
## 'alpha': (-2.4289038916089103, 0.09733311056569602),
## 'cal_217': (0.9628855389985391, 0.012954842207228118),
## 'cal_353': (1.0430339352585394, 0.03414385791991172)}
lkl143_217_3532B50EXTR = pmc.add_lkl(pmc.repar_lkl(pb143_217_3532B50[0],{"cal_143":1,"T":19.6}),("gauss",[-2.42,1.51,1.,1.],[.2**2,.2**2,.05**2,.05**2],["alpha","beta_d","cal_353","cal_217"]))
ch143_217_3532B50EXTR =  pmc.mcmc(lkl143_217_3532B50EXTR,ch143_217_3532B50EXT.meandict(),ch143_217_3532B50EXT.covariance(),nstep=40000)
ch143_217_3532B50EXTR.to_file("ch143_217_353_EE_B50_EXTR.npz")
ch143_217_3532B50EXTR.resdict()
##{'Ddust': (0.0477314464463195, 0.014132662647011338),
## 'beta_d': (1.5604051459168369, 0.16427229211742306),
## 'alpha': (-2.491640374896667, 0.1733070858507529),
## 'cal_100': (1.0097283997318138, 0.03517453207736139),
## 'cal_217': (0.9649171040582396, 0.011737893220019339),
## 'cal_353': (1.0377864426448087, 0.032931727292943094)}


plt.figure(figsize=(cm2inch(10)*1.5*2,cm2inch(10)*1.5))
plt.subplot(121)
ch100_143_217_3532B50EXTR.contour("Ddust","alpha",(0.01,.11,40),(-3.1,-1.9,40),c="k",alpha=.4)
ch143_217_3532B50EXT.contour("Ddust","alpha",(0.02,0.085,40),(-2.8,-2.05,40),c="red")
ch143_217_3532B50.contour("Ddust","alpha",(0.02,0.08,40),(-2.6,-2.2,40),c="blue")
plt.ylabel("$\\alpha_{EE}$")
plt.xlabel("$D_{80}^{\\mathrm{dust}}$")
plt.subplot(122)
ch100_143_217_3532B50EXTR.contour("Ddust","beta_d",(0.01,.11,40),(1.,2.3,40),c="k",alpha=.4)
ch143_217_3532B50EXT.contour("Ddust","beta_d",(0.02,0.085,40),(1.2,1.9,40),c="red")
ch143_217_3532B50.contour("Ddust","beta_d",(0.02,0.085,40),(1.2,1.9,40),c="blue")
plt.axvline(1,c="k",alpha=.4,label="Planck priors $\\times10$")
plt.axvline(2,c="red",label="Planck priors $\\times5$")
plt.axvline(3,c="blue",label="Planck priors $\\times2$")
plt.ylim(1,2.15)
plt.xlim(.01,.105)
plt.ylabel("$\\beta_P$")
plt.xlabel("$D_{80}^{\\mathrm{dust}}$")
plt.legend(frameon=False,fontsize=8)
plt.tight_layout()
plt.savefig("EE_dustXalphaXbeta.pdf")

