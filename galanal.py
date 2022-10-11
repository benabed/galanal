import numpy as nm
import cldf
import numba

class lrange_class:
  def __init__(self,lmin,lmax,binning=None,delta=0,wgh=None):
    self.lmin = lmin
    self.lmax = lmax
    self.bins = None
    if binning is not None:
      self.bins = binning
    elif delta!=0:
      self.add_binning(delta,wgh)

    self.ell = nm.arange(lmin,lmax+1)
    self.lm = self.ell
    if self.bins is not None:
      self.lm = self.bins @ self.ell
    self.llp1 = self.lm * (self.lm+1.)/2./nm.pi
    
  def change_lrange(self,lmin=-1,lmax=-1,binning=None):
    lmin = max(self.lmin,lmin)
    lmax = min(self.lmax,lmax) if lmax>0 else self.lmax
    self.__init__(lmin,lmax,binning)

  def add_binning(self,delta,wgh=None):
    bins = create_binning([self.lmin,self.lmax],delta,wgh)
    self.change_lrange(binning=bins)

def list_outer(fqi,fqj):
    fqc = []
    for qi in fqi:
      for qj in fqj:
        fqc += [(qi,qj)]
    return fqc

def get_fqc(fqi,fqj=None,outer=False,add_kind="",exclude=[],direct=False):  
  # create the list
  if fqj==None:
    fqc = fqi
  else:
    if type(fqi) not in (list,tuple,nm.ndarray):
      fqi = [fqi]
    if type(fqj) not in (list,tuple,nm.ndarray):
      fqj = [fqj]
    if outer==False:
      fqc = zip(fqi,fqj)
    else:
      fqc = list_outer(fqi,fqj)
  # filter the list
  if direct:
    if isinstance(fqc,int):
        return [(fqc,fqc)]
    return fqc

  if len(exclude):
    exclude = get_fqc(exclude,add_kind=add_kind)
  fqqc = []
  for fqi,fqj in fqc:
    if fqj<fqi:
      fqi,fqj = fqj,fqi
    if add_kind:
      if isinstance(fqi,int):
        fqi = str(fqi)
      if fqi[-1] in "0123456789":
        fqi = fqi+"T"
      fqi=fqi[:-1]+add_kind[0]
      if isinstance(fqj,int):
        fqj = str(fqj)
      if fqj[-1] in "0123456789":
        fqj = fqj+"T"
      fqj=fqj[:-1]+add_kind[1]
    if (fqi,fqj) in fqqc or (fqi,fqj) in exclude:
      continue
    fqqc += [(fqi,fqj)]
  return fqqc


class rq:
  jcor = [58.062295,2.2703657]
  def __init__(self,where,nrm=1e12):
    f = cldf.open(where)
    self.data = f["data"]*nrm
    self.data.shape=f["shape"]
    self.lmax = f["lmax"]
    self.TP = nm.array([v=="P" for v in f["pol"]])
    self.nP = nm.sum(self.TP)
    self.nT = len(self.TP)
    self.freq0 = list(f["freq"])
    self.freq = [str(ff)+"T" for ff in self.freq0] + [str(ff)+"E" for ff,pp in zip(self.freq0,self.TP) if pp]    
    for i,fq in enumerate(self.freq):
      nn=-1
      if fq=="545T":
        nn = i 
        jcor = self.jcor[0]
      elif fq=="857T":
        nn = i 
        jcor = self.jcor[1]
      if nn!=-1:
        self.data[nn]/=jcor
        self.data[:,nn]/=jcor
    if "fsky" in f:
      self.fsky = f["fsky"]
    else:
      self.fsky = 1     
    self.lmin = 0
    if "lmin" in f:
      self.lmin=f["lmin"]
    self.ell = nm.arange(self.lmin,self.lmax+1)
    self.llp1 = self.ell*(self.ell+1)/2./nm.pi
    self.fake_data = self.data*1.
    
  def get_lrange(self):
    return lrange_class(self.lmin,self.lmax)

  def get_lcut(self,lrange=None,lmin=-1,lmax=-1,binning=None):
    nlrange = self.get_lrange()
    if lrange is not None:
      lmin=lrange.lmin
      lmax=lrange.lmax
      if lrange.bins is not None and binning is None:
        binning = lrange.bins
    nlrange.change_lrange(lmin,lmax,binning)
    return nlrange

  def get_ell(self,lrange=None,lmin=-1,lmax=-1,binning=None):
    lrange = self.get_lcut(lrange,lmin,lmax,binning)
    return lrange.lm
  
  def get_llp1(self,lrange=None,lmin=-1,lmax=-1,binning=None):
    lrange = self.get_lcut(lrange,lmin,lmax,binning)
    return lrange.llp1
    
  def get_idx(self,fq):
    if fq[-1] in "0123456789":
      fq = fq+"T"
    r = []
    for i,fqi in enumerate(self.freq):
      if fqi==fq:
        r +=[i]
    return r

  def get_spectrum_idx(self,fqi,fqj,sym=True,auto=False):
    if fqi[-1] in "0123456789":
      fqi = fqi+"T"
    if fqj[-1] in "0123456789":
      fqj = fqj+"T"
    ii = self.get_idx(fqi)
    jj = self.get_idx(fqj)
    r = []
    for iia in ii:
      for jja in jj:
        if (not auto) and (iia==jja):
          continue 
        r += [[iia,jja]]
    if sym:
      r += self.get_spectrum_idx(fqi[:-1]+fqj[-1],fqj[:-1]+fqi[-1],sym=False,auto=auto)
    return r

  def _multi_fg(self,fqi,fqj,outer=False):
    if type(fqi) not in (list,tuple,nm.ndarray):
      fqi = [fqi]
    if type(fqj) not in (list,tuple,nm.ndarray):
      fqj = [fqj]
    if outer :
      fqc = []
      for qi in fqi:
        for qj in fqj:
          fqc += [(qi,qj)]  
    else:
      fqc = list(zip(fqi,fqj))
    return fqc

  def get_spectrum(self,fqc,kind="TT",sym=True,auto=False,regul=True,lrange=None,direct=False):
    fqc = get_fqc(fqc,add_kind=kind,direct=direct)
    rspec = []
    for cp in fqc:
      rspec += [self._get_spectrum(cp[0],cp[1],sym,auto,regul,lrange,direct)]
    if len(fqc)==1:
      rspec = rspec[0]
    return nm.array(rspec)

  def _get_spectrum(self,fqi,fqj,sym=True,auto=False,regul=True,lrange=None,direct=False):
    if direct:
      li_idx = [[fqi,fqj]]
      wght = [1.]
    else:
      li_idx = self.get_spectrum_idx(fqi,fqj,sym,auto)
      wght = [self._weight(cs,regul) for cs in li_idx]
    assert len(li_idx),"no spectra %sx%s"%(fqi,fqj)
    cl = [self.data[cs[0],cs[1]]*wgh for cs,wgh in zip(li_idx,wght)]
    cl = nm.sum(cl,0)/nm.sum(wght,0)
    lrange=self.get_lcut(lrange)
    cl = cl[lrange.lmin-self.lmin:lrange.lmax-self.lmin +1]
    if lrange.bins is not None:
      cl = lrange.bins @ cl
    return cl

  def _cheapsig(self,i,j,k,l,regul=True,fake=False):
    if fake:
      vr = self.fake_data[i,k]*self.fake_data[j,l]+self.fake_data[i,l]*self.fake_data[j,k]
    else:
      vr = self.data[i,k]*self.data[j,l]+self.data[i,l]*self.data[j,k]
    nmodes = (2.*nm.arange(self.lmin,self.lmax+1)+1.)*self.fsky
    if regul :
      vr = nm.interp(self.ell,self.ell[vr>0],vr[vr>0])
    return vr/nmodes

  def _weight(self,cs,regul):
    return 1./self._cheapsig(cs[0],cs[1],cs[0],cs[1],regul)

  def _get_var(self,fqi,fqj,cfqi = None,cfqj = None, sym=True,auto=False,regul=True,lrange=None,lmin=-1,lmax=-1,binning=None,fake=False):
    li_idx = self.get_spectrum_idx(fqi,fqj,sym,auto)
    wghtx = [self._weight(cs,regul) for cs in li_idx]
    swghtx = nm.sum(wghtx,0)
    wghtx = [wgx/swghtx for wgx in wghtx]
    if cfqi!=None:
      li_idy = self.get_spectrum_idx(cfqi,cfqj,sym,auto)
      wghty = [self._weight(cs,regul) for cs in li_idx]
      swghty = nm.sum(wghty,0)
      wghty = [wgy/swghty for wgy in wghty]
    else:
      li_idy = li_idx
      wghty = wghtx
      swghty = swghtx
    vr = 0
    for pra,wgha in zip(li_idx,wghtx):
      for prb,wghb in zip(li_idy,wghty):
        vr = vr + wgha*wghb * self._cheapsig(pra[0],pra[1],prb[0],prb[1],regul,fake)
    lrange=self.get_lcut(lrange,lmin,lmax,binning)
    vr = vr[lrange.lmin-self.lmin:lrange.lmax- self.lmin +1 ]
    vr = nm.diag(vr)
    if lrange.bins is not None:
      vr = lrange.bins @ vr @ lrange.bins.T
    return vr
    
  def get_var(self,fqc,kind="TT",sym=True,auto=False,regul=True,outer=False,lrange=None,lmin=-1,lmax=-1,binning=None,fake=False):
    fqc = get_fqc(fqc,add_kind=kind)
    lfqc = len(fqc)
    rvr = [ [ [] for i in range(lfqc)] for j in range(lfqc)]
    for i,cpi in enumerate(fqc):
      vr = self._get_var(cpi[0],cpi[1],sym=sym,auto=auto,regul=regul,lrange=lrange,lmin=lmin,lmax=lmax,binning=binning,fake=fake)
      rvr[i][i]= vr  
      for j in range(0,i):
        cpj = fqc[j]
        vr = self._get_var(cpi[0],cpi[1],cpj[0],cpj[1],sym,auto,regul,lrange,lmin,lmax,binning)
        rvr[i][j]= vr
        rvr[j][i]= vr.T
    return nm.block(rvr)

def create_binning(ls,delta=None,wgh=None,lmin=-1,lmax=-1,norm=True):
  if len(ls)==2:
    _ls = nm.arange(ls[0],ls[1]+1,delta)
    if _ls[-1]!=ls[1]+1:
      _ls = nm.concatenate((_ls,[ls[1]+1]))
    ls = _ls
  if wgh is None:
    wgh = nm.ones((ls[-1]-ls[0]))
  elif isinstance(wgh,str) and wgh.lower=="1/l^2":
    ell = nm.arange(ls[0],ls[-1])
    wgh = 1./(l*(l+1.))
  assert len(wgh) == ls[-1]-ls[0]
  if lmin==-1:
    lmin=ls[0]
  if lmax==-1:
    lmax=ls[-1]-1
  bns = nm.zeros((len(ls)-1,lmax+1-lmin))
  for i in range(len(ls)-1):
    bns[i,ls[i]-lmin:ls[i+1]-lmin]=wgh[ls[i]-ls[0]:ls[i+1]-ls[0]]
    if norm:
      bns[i,ls[i]-lmin:ls[i+1]-lmin] /= nm.sum(bns[i,ls[i]-lmin:ls[i+1]-lmin])
  return bns

@numba.jit
def Bplanck(nu,T,nu0):
  h_sur_kb_GHz = 6.62607015e-34/1.380649e-23*1e9
  return nu**3/nu0**3 * (nm.exp(h_sur_kb_GHz*nu0/T)-1) / (nm.exp(h_sur_kb_GHz*nu/T)-1)

Tref = 19.6
betaref = 1.59

def rescale_dust(nu1,nu2,nu0,T,beta):
  return (nu1*nu2/nu0**2)**(beta-2) * Bplanck(nu1,T,nu0)*Bplanck(nu2,T,nu0) / UsC[nu1]/UsC[nu2]*UsC[nu0]**2
  

def get_U(nu):
  import astropy.units as units
  return 1./(units.thermodynamic_temperature(nu*units.GHz)[0][2](units.brightness_temperature(nu*units.GHz)[0][-1](1)))


C = {100:1.0827732 , 143: 1.0140133 ,217: 1.10993566, 353: 1.10290608,545: 1.09415828,857: 1.01049561}
C_ = {100:1.088, 143:1.017, 217:1.120, 353:1.098}
U_ = {100:0.794, 143:0.592, 217:0.334, 353:0.075}
U = dict([(f,get_U(f)) for f in C])
_UsC = dict([(k,U[k]/C[k]) for k in U])

U150 = get_U(150)

@numba.jit
def slp(lmin,lmax,alpha,lstar):
  ell = nm.arange(lmin,lmax+1)
  #U150 = get_U(150)
  slp = ell**alpha /(lstar**alpha)/lstar/(lstar+1.)*2.*nm.pi *U150**2
  return slp

@numba.jit
def dust(lmin,lmax,fqi,fqj,D80,beta_d,alpha=-2.7,lstar=80,T=19.6,UsC="default"):
  if UsC=="default":
    UsC_nrm = _UsC[fqi]*_UsC[fqj]
  elif isinstance(UsC,dict):
    UsC_nrm = UsC[fqi]*UsC[fqj]
  else:
    UsC_nrm=1
  sl = slp(lmin,lmax,alpha,lstar)
  GAL = D80 * sl * (fqi*fqj/150.**2)**(beta_d-2) * Bplanck(fqi,T,150)* Bplanck(fqj,T,150)/UsC_nrm
  return GAL

def Make_model(kind,cl,lrange,fqc,parnames,**preset):
  ell = lrange.ell
  
  pardict = {
    "Ddust" : 0,
    "beta_d" : 1.5,
    "alpha" : -2.47,
    "lstar" : 80,
    "T" : 19.,
    "gammacib" : 0.51  
  }
  frqs = set()
  for fqi,fqj in fqc:
    fqi = int(fqi)
    fqj = int(fqj)
    frqs.add(fqi)
    frqs.add(fqj)
    pardict["cal_%d"%fqi] = 1.
    pardict["cal_%d"%fqj] = 1.
    pardict["Acib_%d_%d"%(fqi,fqj)] = 0
    pardict["Aps_%d_%d"%(fqi,fqj)] = 0
  frqs = sorted(list(frqs))

  for key in preset:
    if key in pardict:
      pardict[key] = preset[key]

  paruse = []
  for p in parnames:
    if p == "CAL":
      paruse += ["cal_%d"%f for f in frqs]
      continue
    if p in ("CIB","PS"):
      for fqi,fqj in fqc:
        fqi = int(fqi)
        fqj = int(fqj)
        paruse += ["A%s_%d_%d"%(p.lower(),fqi,fqj)]
      continue
    paruse += [p]

  if cl in (None, False, 0):
    cl = nm.zeros(lrange.lmax+1-lrange.lmin)
  if isinstance(cl,str):
    rcl = nm.loadtxt(cl)
    cl = rcl[lrange.lmin-2:lrange.lmax+1-2,{"TT":1,"TE":2,"EE":3}[kind]]/ell/(ell+1.)*2*nm.pi
  
  def Model(pars):
    if isinstance(pars,dict):
      pars = nm.array([pars[k] for k in paruse])
    assert len(pars)==len(paruse)
    #print (paruse,pars)
    pardict.update(dict(zip(paruse,pars)))
    #print(pardict)
    rcl = []
    for fqi,fqj in fqc:
      fqi = int(fqi)
      fqj = int(fqj)
      cal = pardict["cal_%d"%fqi] * pardict["cal_%d"%fqj]
      #print(pardict)
      FG = dust(lrange.lmin,lrange.lmax,fqi,fqj,pardict["Ddust"],pardict["beta_d"],pardict["alpha"],pardict["lstar"],pardict["T"])
      if kind=="TT":
        CIB = pardict["Acib_%d_%d"%(fqi,fqj)]*(lrange.ell/3000.)**pardict["gammacib"]*2*nm.pi/lrange.ell/(lrange.ell+1)
        #print(CIB[-1],"Acib_%d_%d"%(fqi,fqj),pardict["Acib_%d_%d"%(fqi,fqj)],pardict["gammacib"])
        PS = pardict["Aps_%d_%d"%(fqi,fqj)] /3000./3001.*2*nm.pi
        FG = FG + CIB+PS
      pcl = cal * (cl+FG)
      if lrange.bins is not None:
        pcl = lrange.bins @ pcl
      rcl += [pcl]
    return nm.array(rcl)
  Model.paruse = paruse
  return Model

def dust_values(pardict,fqc,ells,cl,kind="TT",fsky=1,UsC="default"):
  res = {}
  if isinstance(ells,(int,float)):
    ells=[ells]
  ells = nm.array(ells)
  ells = nm.sort(ells)
  lmin = ells[0]
  lmax = ells[-1]
  rcl = nm.loadtxt(cl)
  lr = nm.arange(lmin,lmax+50)
  for fqi,fqj in fqc:
    UsC = {fqi:get_U(fqi),fqj:get_U(fqj)}
    dst = dust(lmin,lmax+50,fqi,fqj,pardict["Ddust"],pardict["beta_d"],pardict["alpha"],pardict["lstar"],pardict["T"],UsC = UsC)
    print(dst[0]*80*81/2./nm.pi)
    for el in ells:
      if kind in ("TT","EE"):
        var = 2*rcl[el-lmin-2,{"TT":1,"TE":2,"EE":3}[kind]]**2
      else:
        var = rcl[el-lmin-2,2]**2 + rcl[el-lmin-2,1]*rcl[el-lmin-2,3]
      var = var / ((2*lr+1.)*fsky)/((lr*(lr+1)/2./nm.pi)**2)
      print(el-lmin,el+50-lmin)
      print (nm.sum((var*el*(el+1)*el*(el+1))[el-lmin:el+50-lmin]))
      mres = {
      "Amp Cl":dst[el-lmin],
      "Amp Dl":dst[el-lmin] * el *(el+1)/2./nm.pi,
      "percent CMB": dst[el-lmin] * el *(el+1)/2./nm.pi/rcl[el-lmin-2,{"TT":1,"TE":2,"EE":3}[kind]],
      "percent sig": dst[el-lmin] / (var[el-lmin]**.5),
      "percent sig Delta50": nm.sum((dst*el*(el+1))[el-lmin:el+50-lmin]) / (nm.sum((var*el*(el+1)*el*(el+1))[el-lmin:el+50-lmin]))**.5
      }
      res[(fqi,fqj,el)]=mres
  return res

def make_problem(rq,kind,fqc,lrange,cl,parnames,**preset):
  lrange = rq.get_lcut(lrange)
  data = rq.get_spectrum(fqc,kind,lrange=lrange)
  model = Make_model(kind,cl,lrange,fqc,parnames,**preset)
  model_cmb = Make_model(kind,cl,lrange,fqc,[p for p in parnames if "cal" in p.lower()],**preset)
  model_dust = Make_model(kind,0,lrange,fqc,[p for p in parnames if ("ps" not in p.lower()) and ("cib" not in p.lower())],**preset)
  model_fg = Make_model(kind,0,lrange,fqc,[p for p in parnames if ("ps" in p.lower()) or ("cib" in p.lower())],**preset)
  model_cib = Make_model(kind,0,lrange,fqc,[p for p in parnames if ("cib" in p.lower())],**preset)
  model_ps = Make_model(kind,0,lrange,fqc,[p for p in parnames if ("ps" in p.lower())],**preset)
  var = rq.get_var(fqc,kind,lrange=lrange,fake=preset.get("fake",False))
  err = var.diagonal()**.5
  err.shape=data.shape
  if preset.get("regularize_var",True):
    # regularize the covariance...
    eigv, eigvec = nm.linalg.eig(var.T@var)
    inv_var = eigvec @ nm.diag(1./nm.sqrt(eigv)) @ eigvec.T
  else:
    inv_var = nm.linalg.inv(var)
  def lkl(prs):
    mod = model(prs)
    residual = nm.concatenate(data-mod)
    return -.5 * residual @ (inv_var @ residual)
  lkl.paruse = model.paruse
  return lkl,data,(err,var),(model,model_cmb,model_dust,model_fg,model_cib,model_ps)

def plotpb(pb,kind,fqc,lrange,pars,**extra):
  import pylab as plt
  if "fig" not in  extra:
    fig = plt.figure()
  else:
    if extra["fig"]==-1:
      pass
    else:
      fig = plt.figure(extra["fig"])
  clrs = plt.get_cmap("tab20").colors
  datpts = []
  oneCMB=False
  for i,fq in enumerate(fqc):
    if "exclude" in extra:
      if i in extra["exclude"] or fq in extra["exclude"]:
        continue
    if "only" in extra:
      extra["only"] = [tuple(v) for v in extra["only"]]
      print(extra["only"])
      print(i,fq)
      print((i,) in extra["only"],fq in extra["only"])
      if (i,) in extra["only"] or fq in extra["only"]:
        pass
      else:
        continue
    label  ="%d%s\\times%d%s"%(fq[0],kind[0],fq[1],kind[1])
    aa=plt.errorbar(lrange.lm,pb[1][i]*lrange.llp1,pb[2][0][i]*lrange.llp1,label="$%d%s\\times%d%s$"%(fq[0],kind[0],fq[1],kind[1]),marker="o",lw=.5,ls="",c=clrs[2*i+1])
    datpts +=[aa]
    tot = plt.loglog(lrange.lm,pb[3][0](pars)[i]*lrange.llp1,lw=2,c=clrs[2*i],label="total")
    if not oneCMB:
      cmb, = plt.loglog(lrange.lm,pb[3][1](pars)[i]*lrange.llp1,c="k",lw=2,label="cmb")
      oneCMB = True
    dst, = plt.loglog(lrange.lm,pb[3][2](pars)[i]*lrange.llp1,lw=2,ls="--",c=clrs[2*i],label="dust")
    if "cib" in " ".join(pb[3][0].paruse).lower():
      fg, = plt.loglog(lrange.lm,pb[3][3](pars)[i]*lrange.llp1,lw=1,ls="-.",c=clrs[2*i],label = "diffuse fg")
      cib, = plt.loglog(lrange.lm,pb[3][4](pars)[i]*lrange.llp1,lw=.5,ls="-.",c=clrs[2*i+1], label="cib")
    #fig.loglog(lrange.lm,pb[3][5](pars)[i]*lrange.llp1,color=aa[0].get_color(),lw=.5,ls="-.")
  elgd = []
  if "cib" in " ".join(pb[3][0].paruse).lower():
    elgd = [fg,cib]
  plt.legend(handles= [cmb,dst] + elgd +datpts,frameon=False,fontsize=8,ncol=3,loc=extra.get("loclegend",0))
  ax = plt.gca()
  leg = ax.get_legend()
  leg.legendHandles[1].set_color('grey')
  if len(elgd):
    leg.legendHandles[2].set_color('grey')
    leg.legendHandles[3].set_color('grey')

  plt.tight_layout()

def create_fake_data(rq,kind,fqc,lrange,delta_noise,cl,parname,bestfit,**preset):
  lrange_0 = lrange_class(lrange.lmin,lrange.lmax)
  lrange_noise = rq.get_lcut(lmin=lrange.lmin,lmax=lrange.lmax+1000)
  lrange_noise.add_binning(delta=delta_noise)
  model = Make_model(kind,cl,lrange_0,fqc,parname,**preset)
  fqc = get_fqc(fqc,add_kind=kind)
  done = []
  fake = rq.fake_data
  for i,cp in enumerate(fqc):
    li_idx = rq.get_spectrum_idx(cp[0],cp[1],sym=True,auto=True)
    for cs in li_idx:
      if cs in done:
        continue
      done +=[cs]
      fake[cs[0],cs[1],lrange_0.lmin-rq.lmin:lrange_0.lmax+1-rq.lmin] = model(bestfit)[i]
      fake[cs[1],cs[0],lrange_0.lmin-rq.lmin:lrange_0.lmax+1-rq.lmin] = model(bestfit)[i]
      if cs[0]==cs[1]:
        noise = nm.interp(lrange_0.lm,lrange_noise.lm,rq.get_spectrum(cs[0],lrange=lrange_noise,direct=True)-rq.get_spectrum([cp],lrange=lrange_noise))
        fake[cs[0],cs[1],lrange_0.lmin-rq.lmin:lrange_0.lmax+1-rq.lmin] += noise



      





