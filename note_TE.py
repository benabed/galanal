import numpy as nm
import galanal as gnl
import pmclite as pmc

rq = gnl.rq("data/rq_spt_winter.cldf/")
lrangeB30 = gnl.lrange_class(100,1500,delta=30)
lrangeB49 = gnl.lrange_class(100,1500,delta=49)
lrangeB50 = gnl.lrange_class(100,1500,delta=50)
lrangeB80 = gnl.lrange_class(100,1500,delta=80)
cl = "attic/base_plikHM_TTTEEE_lowl_lowE_lensing.minimum.theory_cl"

parbase_T = ["beta_d","Ddust","alpha","T","CAL"]
fqc = gnl.get_fqc([143,217,353],[217,353,545],outer=True,exclude=([[143,217],[143,353]]))
fqc_no545 = gnl.get_fqc([143,217,353],[217,353],outer=True,exclude=([[143,217],[143,353]]))

pb143_217_353_545B30 = gnl.make_problem(rq,"TE",fqc,lrangeB30,cl,parbase_T)
#pb143_217_353_545B49 = gnl.make_problem(rq,"TE",fqc,lrangeB49,cl,parbase_T)
pb143_217_353_545B50 = gnl.make_problem(rq,"TE",fqc,lrangeB50,cl,parbase_T)
pb143_217_353_545B80 = gnl.make_problem(rq,"TE",fqc,lrangeB80,cl,parbase_T)
pb143_217_353B50 = gnl.make_problem(rq,"TE",fqc_no545,lrangeB50,cl,parbase_T)

lkl143_217_353_545B30 = pmc.add_lkl(pmc.repar_lkl(pb143_217_353_545B30[0],{"cal_143":1,"cal_545":1,"T":19.6}),("gauss",[-2.42,1.51,1.,1.],[.04**2,.04**2,.02**2,.02**2],["alpha","beta_d","cal_353","cal_217"]))
#lkl143_217_353_545B49 = pmc.add_lkl(pmc.repar_lkl(pb143_217_353_545B49[0],{"cal_143":1,"cal_545":1,"T":19.6}),("gauss",[-2.42,1.51,1.,1.],[.04**2,.04**2,.02**2,.02**2],["alpha","beta_d","cal_353","cal_217"]))
lkl143_217_353_545B50 = pmc.add_lkl(pmc.repar_lkl(pb143_217_353_545B50[0],{"cal_143":1,"cal_545":1,"T":19.6}),("gauss",[-2.42,1.51,1.,1.],[.04**2,.04**2,.02**2,.02**2],["alpha","beta_d","cal_353","cal_217"]))
lkl143_217_353_545B80 = pmc.add_lkl(pmc.repar_lkl(pb143_217_353_545B80[0],{"cal_143":1,"cal_545":1,"T":19.6}),("gauss",[-2.42,1.51,1.,1.],[.04**2,.04**2,.02**2,.02**2],["alpha","beta_d","cal_353","cal_217"]))

lkl143_217_353B50 = pmc.add_lkl(pmc.repar_lkl(pb143_217_353B50[0],{"cal_143":1,"cal_545":1,"T":19.6}),("gauss",[-2.42,1.51,1.,1.],[.04**2,.04**2,.02**2,.02**2],["alpha","beta_d","cal_353","cal_217"]))
lkl143_217_353_545B50EXT = pmc.add_lkl(pmc.repar_lkl(pb143_217_353_545B50[0],{"cal_143":1,"cal_545":1,"T":19.6}),("gauss",[-2.42,1.51,1.,1.],[.1**2,.1**2,.05**2,.05**2],["alpha","beta_d","cal_353","cal_217"]))
lkl143_217_353_545B50EXTR = pmc.add_lkl(pmc.repar_lkl(pb143_217_353_545B50[0],{"cal_143":1,"cal_545":1,"T":19.6}),("gauss",[-2.42,1.51,1.,1.],[.2**2,.2**2,.05**2,.05**2],["alpha","beta_d","cal_353","cal_217"]))

Zh143_217_353_545 = pmc.chain("ch143_217_353_545_TE.npz")
ch143_217_353_545B30 =  pmc.mcmc(lkl143_217_353_545B30,Zh143_217_353_545.meandict(),Zh143_217_353_545.covariance(),nstep=40000)
ch143_217_353_545B30.to_file("ch143_217_353_545_TE_B30.npz")

ch143_217_353_545B50 =  pmc.mcmc(lkl143_217_353_545B50,ch143_217_353_545B30.meandict(),ch143_217_353_545B30.covariance(),nstep=5000)
ch143_217_353_545B50.to_file("ch143_217_353_545_TE_B50.npz")

ch143_217_353_545B49 =  pmc.mcmc(lkl143_217_353_545B49,ch143_217_353_545B30.meandict(),ch143_217_353_545B30.covariance(),nstep=5000)
ch143_217_353_545B49.to_file("ch143_217_353_545_TE_B49.npz")

ch143_217_353_545B80 =  pmc.mcmc(lkl143_217_353_545B80,ch143_217_353_545B30.meandict(),ch143_217_353_545B30.covariance(),nstep=5000)
ch143_217_353_545B80.to_file("ch143_217_353_545_TE_B80.npz")

ch143_217_35350 =  pmc.mcmc(lkl143_217_35350,ch143_217_353_545B30.meandict(),ch143_217_353_545B30.covariance(),nstep=5000)
ch143_217_353B50.to_file("ch143_217_353_TE_B50.npz")

ch143_217_353_545B50EXT =  pmc.mcmc(lkl143_217_353_545B50EXT,Zh143_217_353_545.meandict(),ch143_217_353_545B50EXT.covariance(),nstep=40000)
ch143_217_353_545B50EXT.to_file("ch143_217_353_545_TE_B50EXT.npz")

ch143_217_353_545B50EXTR =  pmc.mcmc(lkl143_217_353_545B50EXTR,Zh143_217_353_545.meandict(),ch143_217_353_545B50EXTR.covariance(),nstep=40000)
ch143_217_353_545B50EXTR.to_file("ch143_217_353_545_TE_B50EXTR.npz")

ch143_217_353_545B30 = pmc.chain("ch143_217_353_545_TE_B30.npz")
ch143_217_353_545B50 = pmc.chain("ch143_217_353_545_TE_B50.npz")
ch143_217_353_545B80 = pmc.chain("ch143_217_353_545_TE_B80.npz")
ch143_217_353B50 = pmc.chain("ch143_217_353_TE_B50.npz")
ch143_217_353_545B50EXT = pmc.chain("ch143_217_353_545_TE_B50EXT.npz")
ch143_217_353_545B50EXTR = pmc.chain("ch143_217_353_545_TE_B50EXTR.npz")


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

plt.figure(figsize=(cm2inch(10)*1.5,cm2inch(10)*1.5))
gnl.plotpb(pb143_217_353_545B50,"TE",fqc,lrangeB50,ch143_217_353_545B50EXT.meandict()|{"cal_143":1,"T":19.6,"cal_545":1},fig=-1,loclegend=3,only=([353,545],[217,545],[353,353],[217,217]))
plt.yscale("linear")
plt.ylim(-180,1100)
plt.ylabel("$\\mathcal{D}_\ell\ [\mu K^2]$")
plt.xlabel("$\ell$")
plt.title("$D^{\mathrm{dust}}_{80} = %.3g \pm %.2g, \\alpha =  %.3g \pm %.2g, \\beta = %.3g \pm %.2g$"%(ch143_217_353_545B50.meandict()["Ddust"],ch143_217_353_545B50.stddict()["Ddust"],ch143_217_353_545B50.meandict()["alpha"],ch143_217_353_545B50.stddict()["alpha"],ch143_217_353_545B50.meandict()["beta_d"],ch143_217_353_545B50.stddict()["beta_d"]))
plt.tight_layout()
plt.savefig("BF_143_217_353_545_TE_clB50.pdf")

clrs = plt.get_cmap("tab20c").colors
plt.figure(figsize=(cm2inch(10)*1.5*2,cm2inch(10)*1.5))
plt.subplot(121)
ch143_217_353_545B50EXTR.contour("alpha","Ddust",(-2.8,-2.1,40),(0.02,0.3,40),c=clrs[16],alpha=.3)
ch143_217_353B50.contour("alpha","Ddust",(-2.6,-2.2,40),(0.02,0.24,40),c=clrs[12],alpha=.8,clabel=False)
ch143_217_353_545B50EXT.contour("alpha","Ddust",(-2.8,-2.1,40),(0.02,0.3,40),c=clrs[4],alpha=.6,clabel=False)
ch143_217_353_545B50.contour("alpha","Ddust",(-2.6,-2.2,40),(0.05,0.2,40),c=[clrs[2],clrs[3]],fill=True,clabel=False,cross=False, lines=False)
ch143_217_353_545B50.contour("alpha","Ddust",(-2.6,-2.2,40),(0.05,0.2,40),c=clrs[0],fill=False)
plt.xlabel("$\\alpha_{EE}$")
plt.ylabel("$D_{80}^{\\mathrm{dust}}$")
plt.subplot(122)
ch143_217_353_545B50EXT.contour("beta_d","Ddust",(1.,2,40),(0.02,0.3,40),c=clrs[4],alpha=.6,clabel=False)
ch143_217_353_545B50EXTR.contour("beta_d","Ddust",(1.,2,40),(0.02,0.3,40),c=clrs[16],alpha=.3)
ch143_217_353B50.contour("beta_d","Ddust",(1.,2.,40),(0.02,0.3,40),c=clrs[12],alpha=.8,clabel=False)
ch143_217_353_545B50.contour("beta_d","Ddust",(1.2,1.8,40),(0.05,0.2,40),c=[clrs[2],clrs[3]],fill=True,clabel=False,cross=False, lines=False)
ch143_217_353_545B50.contour("beta_d","Ddust",(1.2,1.8,40),(0.05,0.2,40),c=clrs[0],fill=False)
plt.axvline(1,c=clrs[16],alpha=.3,label="Planck priors $\\times10$")
plt.axvline(2,c=clrs[4],alpha=.6,label="Planck priors $\\times5$")
plt.axvline(2,c=clrs[12],alpha=.8,label="no 545")
plt.axvline(3,c=clrs[0],label="Planck priors $\\times2$")
plt.xlim(1.2,2.)
plt.ylim(.02,.3)
plt.xlabel("$\\beta_P$")
plt.ylabel("$D_{80}^{\\mathrm{dust}}$")
plt.legend(frameon=False,fontsize=8)
plt.tight_layout()
plt.savefig("ET_dustXalphaXbeta.pdf")
plt.savefig("ET_dustXalphaXbeta.png")


