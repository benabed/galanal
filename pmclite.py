import numpy as nm

import numpy.random as ra

def mask_and_order(lkl,mask_and_order,moy,sig=None):
  if any(mask_and_order<0):
    order = mask_and_order
  else:
    order = []
    j=0
    for m in mask_and_order:
      if m>0:
        order +=[j]
        j+=1
      else:
        order +=[-1]
  ntot = len(mask_and_order)
  nnew = 0
  mx = -1
  for i in range(ntot):
    if order[i]>=0:
      nnew+=1
      mx = max(mx,order[i])
  print (nnew,mx,order)
  assert nnew==mx+1

  def mlkl(pars):
    npars = moy*1.
    for i in range(ntot):
      if order[i]!=-1:
        npars[i] = pars[order[i]]
    return lkl(npars)
  osig = sig
  if sig is None:
    osig = nm.zeros((ntot,ntot))
  nmoy = nm.zeros(nnew)
  nsig = nm.zeros((nnew,nnew))
  for i in range(ntot):
    if order[i]>=0:
      nmoy[order[i]]=moy[i]
      nsig[order[i],order[i]]=osig[i,i]
      for j in range(i+1,ntot):
        if order[j]>=0:
          nsig[order[i],order[j]]=osig[i,j]
          nsig[order[j],order[i]]=osig[j,i]
  if sig is None:
    return mlkl,moy
  return  mlkl,nmoy,nsig

def pmc(lkl,moy,sig,N):
  moy = nm.array(moy)
  sig = nm.array(sig)
  tb = ra.multivariate_normal(moy,sig,N)
  lv = nm.array([lkl(v) for v in tb]).flat[:]
  gv = -nm.dot(nm.dot((tb-moy[nm.newaxis,:]),nm.linalg.inv(sig)),(tb-moy[nm.newaxis,:]).T).diagonal()/2.
  w = lv-gv
  w = w-w.max()
  w = nm.exp(w)
  w = w/nm.sum(w)
  print("ESS",1./nm.sum(w**2))
  print("perp",nm.exp(-nm.sum(w*nm.log(w)))/N)
  return w,tb

def multi_pmc(lkl,moy,sig,N,m,NL,thr=.96):
  for i in range(m):
    print (moy,nm.sqrt(sig.diagonal()))
    w,tb = pmc(lkl,moy,sig,N)
    moy,sig=moy_std(w,tb,False)
    if nm.exp(-nm.sum(w*nm.log(w)))/N>thr:
      break
  return pmc(lkl,moy,sig,NL)

def moy_std(w,tb,d=True):
  moy = nm.sum(tb*w[:,nm.newaxis],0)/w.sum()
  sig = varme(w,tb,moy)
  if d:
    sig = sig.diagonal()
  return moy,sig

def varme(w,tb,moy=None):
  if moy is None:
    moy = nm.sum(tb*w[:,nm.newaxis],0)/w.sum()
  mtb = tb - moy[nm.newaxis,:]

  return  nm.sum([nm.outer(mtb[i],mtb[i])*w[i] for i in range(mtb.shape[0])],0)/w.sum()

def contourme(wtb,i,j,ibins,jbins,c="k",cross=True,lines=True,legend=False,*args,**kargs):
  import pylab as plt
  w,tb = wtb
  ib,jb,iarr = regrid2d(w,tb,i,j,ibins,jbins)
  if "colors" not in kargs:
    kargs["colors"]=c
  pretty_contour(ib,jb,iarr.T,*args,**kargs)
  moy,std = moy_std(w,tb)
  if cross:
    if legend:
      plt.plot((moy[i],),(moy[j],),marker="+",ms="5",mew="2",c=c,label=legend,lw=0)
    else:
      plt.plot((moy[i],),(moy[j],),marker="+",ms="5",mew="2",c=c)
  if lines: 
    plt.axvline((moy[i],),lw=.5,c=c)
    plt.axhline((moy[j],),lw=.5,c=c)

def minmax(tb):
  return nm.array((nm.min(tb,0),nm.max(tb,0)))
def minmax_range(tb,i,ni):
    mima = minmax(tb)
    return nm.linspace(mima[0,i],mima[1,i],ni)

def regrid2d(w,tb,i,j,ibins,jbins):
  """ compute a 2d approximation using a kde for the ith and jth dimensions, suitable for a contour plot"""
  import scipy.stats.kde as kde
  dens = kde.gaussian_kde([tb[:,i],tb[:,j]],weights=w)
  if type(ibins)==int:
    ibins = minmax_range(tb,i,ibins)
  elif len(ibins)in (2,3):
    nbins = ibins[2] if len(ibins)==3 else 30
    ibins = nm.linspace(ibins[0],ibins[1],nbins)
  if type(jbins)==int:
    jbins = minmax_range(tb,j,jbins)
  elif len(jbins)in (2,3):
    nbins = jbins[2] if len(jbins)==3 else 30
    jbins = nm.linspace(jbins[0],jbins[1],nbins)

  mcarr = gridme(ibins,jbins,dens)
  return ibins,jbins,mcarr

def gridme(p0,p1,func,loglkl=False):
    """ Tabulate a function evaluated on a 2d-grid defined by arrays p0 and p1"""
    lkarr = nm.zeros((len(p0),len(p1)))
    for i,pp in enumerate(p0):
        lkarr[i,:]=[func([pp,py]) for py in p1]
    if loglkl:
      mx = nm.max(lkarr)
      lkarr = nm.exp((lkarr-mx))
    return lkarr



def _get_level(prob_arr,v):
 # première étape, reclasser une version monodimensionnelle de la grille par ordre décroissant.
  sort_grid = prob_arr.flat[:]*1.
  sort_grid.sort() # en ordre croissant
  sort_grid = sort_grid[::-1] # en ordre decroissant

  # on calcule maintenant la cumulative
  cis=nm.cumsum(sort_grid)
  nrm = cis[-1] # on calcule la normalisation
  cis/=nrm # et on renormalise
  # la fonction searchsorted renvoie l'index ou se situerai un nombre dans un tableau ordonné
  # cis est ordonné par construction
  a_pv = nm.searchsorted(cis,v) 
  # a_pv est l'index pour lequel cis=v, 
  # c'est aussi l'index de la dernière valeur ajoutée pour estimer la cumulative
  levl = sort_grid[a_pv]
  return levl

def pretty_contour(x,y,z,clabel=True,get_contour_levels=_get_level, levels=(0.68,0.95,0.99),fill=False,*args,**kargs):
  import pylab as plt
  """ Plot 2d contours of input array z containing 68%, 95% and 99% of the points """
  lvls = [get_contour_levels(z,ll) for ll in levels]
  dvls = dict(zip(lvls,["%.0f\\%%"%(v*100) for v in levels]))
  if "colors" not in kargs:
    kargs["colors"]="black"
  if fill:
    CS=plt.contourf(x, y, z,lvls[::-1],*args,**kargs)
  else:  
    CS=plt.contour(x, y, z,lvls[::-1],*args,**kargs)
  if clabel:
    plt.gca().clabel(CS, fmt=dvls,inline=1, fontsize=7)

def join_chain(ch1,ch2):
  chj = chain(ch1.chain[0],ch1.lkl[0])
  chj.chain = nm.concatenate((ch1.chain[:ch1.nval],ch2.chain[:ch2.nval]))
  chj.lkl = nm.concatenate((ch1.lkl[:ch1.nval],ch2.lkl[:ch2.nval]))
  chj.weight = nm.concatenate((ch1.weight[:ch1.nval],ch2.weight[:ch2.nval]))
  chj.nstep = ch1.nstep+ch2.nstep
  chj.nval = ch1.nval+ch2.nval
  chj.maxval = ch1.maxval+ch2.maxval
  chj.ndim = ch1.ndim
  chj.paruse = ch1.paruse
  return chj





class chain:
  # une classe simple pour une chaine mcmc
  def __init__(self,pos,lklf=None,paruse=[]):
    if lklf==None:
      assert isinstance(pos,str)
      rr = nm.load(pos)
      self.nstep = int(rr["nstep"]) # nombre de pas 
      self.nval = int(rr["nval"]) # nombre de pas differents nval<=nstep
      self.maxval=int(rr["maxval"]) # taille du buffer
      self.weight = rr["weight"] # poids de chaque pas different. nstep = sum(weight)
      self.chain = rr["chain"] # positions dans l'espace des parametres
      self.lkl = rr["lkl"] # log vraisemblance à chaque position
      self.ndim=self.chain.shape[1] # nombre de dimension de l'espace des paramètres
      self.paruse = list(rr["paruse"])
    else:
      self.nstep = 0 # nombre de pas 
      self.nval = 0 # nombre de pas differents nval<=nstep
      self.maxval=1000 # taille du buffer
      self.ndim=len(pos) # nombre de dimension de l'espace des paramètres
      self.weight = nm.zeros(self.maxval) # poids de chaque pas different. nstep = sum(weight)
      self.chain = nm.zeros((self.maxval,self.ndim)) # positions dans l'espace des parametres
      self.lkl = nm.zeros(self.maxval) # log vraisemblance à chaque position
      self.accept(pos,lklf)
      if paruse:
        self.paruse = paruse

  def to_file(self,filename):
    nm.savez(filename,chain=self.chain,weight=self.weight,lkl=self.lkl,nstep=self.nstep,nval=self.nval,maxval=self.maxval,paruse=self.paruse)
  def accept(self,pos,lklf):
    if self.nval==self.maxval:
      weight = nm.zeros(self.maxval*2) # poids de chaque pas different. nstep = sum(weight)
      chain = nm.zeros((self.maxval*2,self.ndim)) # positions dans l'espace des parametres
      lkl = nm.zeros(self.maxval*2) # log vraisemblance à chaque position
      weight[:self.maxval] = self.weight
      chain[:self.maxval] = self.chain
      lkl[:self.maxval] = self.lkl
      self.weight = weight
      self.chain = chain
      self.lkl = lkl
      self.maxval*=2
    self.weight[self.nval] = 1
    self.lkl[self.nval] = lklf
    self.chain[self.nval] = pos
    self.nval+=1
    self.nstep+=1
  def reject(self):
    self.weight[self.nval-1] += 1
    self.nstep+=1

  def integrate(self,func):
    return nm.sum(self.weight*[func(self.chain[i]) for i in range(self.nval)])/self.nstep
  
  #compute mean of chain
  def mean(self):  
    return nm.sum(self.chain[:self.nval]*self.weight[:self.nval,nm.newaxis],axis=0)/self.nstep
    # can also use nm.average(chain[:,2:],weigths=chain[:,0])
  def meandict(self):
    me = self.mean()
    return dict(zip(self.paruse,me))
  def stddict(self):
    me = self.covariance()
    return dict(zip(self.paruse,me.diagonal()**.5))
  def resdict(self):
    md = self.meandict()
    sd = self.stddict()
    for k in md:
      md[k] = (md[k],sd[k])
    return md


  #compute covariance of chain
  def covariance(self):
    mean = self.mean()
    # could do without a loop, but that would create an unnecessary large array to be reduced !
    res = nm.zeros((self.ndim,self.ndim))
    normalization = 1.*self.nstep
    pars0 = self.chain[:self.nval] - mean[nm.newaxis,:]
    for i in range(self.ndim):
      res[i,i] = nm.sum(pars0[:,i]*pars0[:,i]*self.weight[:self.nval],axis=0)/normalization
      for j in range(i+1,self.ndim):
        res[i,j] = nm.sum(pars0[:,i]*pars0[:,j]*self.weight[:self.nval],axis=0)/normalization
        res[j,i] = res[i,j]
    return res
    # can also use nm.cov(chain[:,2:],fweights=chain[:,0])

  # compute acceptance
  def acceptance(self):
    return (1.*self.nval)/self.nstep

  def minmax(self):
    #return range for parameters
    return nm.array([nm.min(self.chain[:self.nval],axis=0),nm.max(self.chain[:self.nval],axis=0)])

  def minmax_range(self,i,ni):
    mima = self.minmax()
    return nm.linspace(mima[0,i],mima[1,i],ni)


  def regrid2d(self,i,j,ibins,jbins):
    """ compute a 2d approximation using a kde for the ith and jth dimensions, suitable for a contour plot"""
    import scipy.stats.kde as kde
    dens = kde.gaussian_kde([self.chain[:self.nval,i],self.chain[:self.nval,j]],weights=self.weight[:self.nval])
    if type(ibins)==int:
      ibins = self.minmax_range(i,ibins)
    elif len(ibins) in (2,3):
      nbins = ibins[2] if len(ibins)==3 else 30
      ibins = nm.linspace(ibins[0],ibins[1],nbins)
    if type(jbins)==int:
      jbins = self.minmax_range(j,jbins)
    elif len(jbins) in (2,3):
      nbins = jbins[2] if len(jbins)==3 else 30
      ibins = nm.linspace(jbins[0],jbins[1],nbins)
    mcarr = gridme(ibins,jbins,dens)
    return ibins,jbins,mcarr

  def plot_brown(self,i,j):
    plt.plot(self.chain[:self.nval,i],self.chain[:self.nval,j],marker=".",ms=4,lw=.7,c="lightblue",alpha=.4,zorder=-100)

  def contour(self,i,j,ibins,jbins,c="k",cross=True,lines=True,legend=False,fill=False,*args,**kargs):
    import pylab as plt
    if isinstance(i,str):
      i = self.paruse.index(i)
    if isinstance(j,str):
      j = self.paruse.index(j)
    w = self.weight
    tb = self.chain
    ib,jb,iarr = regrid2d(w,tb,i,j,ibins,jbins)
    if "colors" not in kargs:
      kargs["colors"]=c
    pretty_contour(ib,jb,iarr.T,fill=fill,*args,**kargs)
    moy = self.mean()
    if cross:
      if legend:
        plt.plot((moy[i],),(moy[j],),marker="+",ms="5",mew="2",c=c,label=legend,lw=0)
      else:
        plt.plot((moy[i],),(moy[j],),marker="+",ms="5",mew="2",c=c)
    if lines: 
      plt.axvline((moy[i],),lw=.5,c=c)
      plt.axhline((moy[j],),lw=.5,c=c)

def r_sig(sig):
  sig = nm.array(sig)
  if len(sig.shape)==1:
    sig = nm.diag(sig)
  return sig * (2.38**2)/sig.shape[0]

# mcmc routine
def mcmc(lkl,xi,SigmaProp,nstep=1000,extra=(),rescale_sig = True, progress=True,print_acc_step=-10):
  """ 
  lkl is the log lkl function
  xi is the starting point in the parameters space
  SigmaProp is the covariance matrix used to propose a new point
  """
  if isinstance(xi,dict):
    xi = nm.array([xi[p] for p in lkl.paruse])
  xi = nm.array(xi)
  lkli = lkl(xi) # compute init lkl
  paruse = None
  if "paruse" in dir(lkl):
    paruse = lkl.paruse
  ch = chain(xi,lkli,paruse)
  if rescale_sig:
    SigmaProp = r_sig(SigmaProp)
  _range = range
  _print = print
  if progress:
    import tqdm
    _range = tqdm.trange
    _print=tqdm.tqdm.write
  if print_acc_step!=0:
    if print_acc_step<0:
      print_acc_step = int(nm.floor(nstep/-print_acc_step))
    elif print_acc_step<1:
      print_acc_step = int(nm.floor(nstep*print_acc_step))
  print("starting")
  for i in _range(nstep):
    ac,lklf,xf = mcmc_step(lkl,xi,SigmaProp,extra,lkli) # do one step
    if ac: 
      ch.accept(xf,lklf)
      lkli = lklf
      xi = xf      
    else: # refuse !
      ch.reject()
    if i%print_acc_step==0 and i!=0:
      _print("step %d - acceptance %g"%(i,ch.acceptance()))
  return ch

def lkl2chi2(func):
  return lambda prs:-2*func(prs)

def mcmc_step(lkl,xi,SigmaProp,extra=(),lkl_0=None):
  """
  lkl is the likelihood function
  xi is the current point in the parameters space
  SigmaProp is the covariance matrix used to propose a new point
  """

  if lkl_0==None:          
    lkl_0 = lkl(xi,*extra) # compute the init log likelihood if needed

  #### --------------------------------------
  #### Exercice : propose a new point here, using xi and SigmaProp 
  # xf = 
  #### --------------------------------------
  xf = nm.random.multivariate_normal(xi,SigmaProp,1)[0] # propose a new point
  try:
    lkl_1 = lkl(xf,*extra)
  except:
    return False,lkl_0,xi  # gracefully refuse the point if it's out of the acceptable parameters range

  #### --------------------------------------
  #### Exercice : Check acceptance here !
  # accept = SOMECONDITION (boolean)
  #### --------------------------------------
  #If chi2 
  #accept = nm.random.uniform()<nm.exp(-.5*(chi2_1-chi2_0))
  # if log lkl
  accept = nm.random.uniform()<=nm.exp((lkl_1-lkl_0))
  
  if accept:    # check acceptance 
    return True,lkl_1,xf # accept
  return False,lkl_0,xi # refuse

def make_lkl_banana(prs0,sig,banane):
  # generic gaussian unnormalized log likelihood
  if len(sig.shape)==1:
    sig = nm.diag(sig)
  siginv = nm.linalg.inv(sig)
  prs0 = nm.array(prs0)
  def lkl(prs):
    pp = prs-prs0
    pp[1]=pp[1]-banane*(pp[0]**2-sig[0,0])
    return -.5*nm.dot(pp,nm.dot(siginv,pp)) # log-likelihood : likelihood = exp(-0.5 chi^2)
  return lkl

def array_or_dict(pars,paruse):
  if isinstance(pars, dict):
    pars = nm.array([pars[p] for p in paruse])
  return nm.array(pars)
    
def accept_dict(paruse):
  def accdic(lkl):
    def nlkl(pars):
      pars = array_or_dict(pars,paruse)
      return lkl(pars)
    nlkl.paruse=paruse
    return nlkl
  return accdic


def make_gauss(prs0,sig,paruse=[]):
  prs0 = nm.array(prs0)
  sig = nm.array(sig)
  if len(sig.shape)==1:
    sig = nm.diag(sig)
  if len(prs0)==1:
    def glkl(prs):
      return -.5*(prs[0]-prs0[0])**2/sig[0,0]
  else:
    glkl = make_lkl_banana(prs0,sig,0)
  if paruse:
    return accept_dict(paruse)(glkl)
  return glkl

def make_uniform(prsmin,prsmax,paruse=[]):
  def unif(pars):
    assert nm.all((pars>=prsmin) * (pars<=prsmax))
    return 0
  if paruse:
    return accept_dict(paruse)(unif)
  return unif

def add_lkl(*lkl_list):
  all_lkl = []
  paruse = []
  for lkl in lkl_list:
    #print(lkl)
    if isinstance(lkl,(list,tuple)):
      if lkl[0][0].lower()=="g":
        all_lkl += [make_gauss(lkl[1],lkl[2],lkl[3])]
      else:
        all_lkl += [make_uniform(lkl[1],lkl[2],lkl[3])]
    else:
      all_lkl += [lkl]
    lparuse = all_lkl[-1].paruse
    paruse += [p for p in lparuse if p not in paruse]
    
  def nlkl(pars):
    
    pdict = dict(zip(paruse,pars))
    res = 0
    for lkl in all_lkl:
      res += lkl(pdict)
    return res
  return accept_dict(paruse)(nlkl)

def add_mean_sig(paruse,*list_ms):
  sig = nm.zeros((len(paruse),len(paruse)))
  pos = nm.zeros((sig.shape[0],))
  for ms in list_ms:
    lparuse = ms[0]
    ip = [paruse.index(p) for p in lparuse]
    pos[ip]=ms[1]
    lsig = nm.array(ms[2])
    if len(lsig.shape)==1:
      lsig = nm.diag(sig)
    for i in range(len(ms[1])):
      for j in range(len(ms[1])):
        sig[ip[i],ip[j]]=lsig[i,j]
  return pos,sig

def add_sig(paruse,*list_ms):
  sig = nm.zeros((len(paruse),len(paruse)))
  for ms in list_ms:
    lparuse = ms[0]
    ip = [paruse.index(p) if p in paruse else -1 for p in lparuse]
    lsig = nm.array(ms[1])
    if len(lsig.shape)==1:
      lsig = nm.diag(lsig)
    for i in range(lsig.shape[0]):
      if ip[i]==-1:
        continue
      for j in range(lsig.shape[1]):
        if ip[j]==-1:
          continue
        sig[ip[i],ip[j]]=lsig[i,j]
  return sig

def combine_lkl(lkl_list):
  """
  Renvoie une fonction qui est la somme 
  des fonctions de log-vraisemblance fournies dans la liste lkl_list
  """
  def nlkl(prs):
    lkv = 0
    for lkl in lkl_list:
      select = False    
      if isinstance(lkl,(list,tuple)):
        tlkl = lkl
        lkl = tlkl[0]
        dic = tlkl[1]
        if "select" in dic:
          select = list(dic["select"])
      if select:
        lkv += lkl(nm.array(prs)[select])
      else:
        lkv += lkl(prs)
    return lkv
  return nlkl

def repar_lkl(lkl,fix={}):
  paruse = [k for k in lkl.paruse if k not in fix]
  def nlkl(pars):
    if isinstance(pars,dict):
      pdict = dict(zip(paruse,[pars[p] for p in paruse]))
    else:
      pdict = dict(zip(paruse,[p for p in pars]))
    pdict.update(fix)
    return lkl(pdict)
  nlkl.paruse=paruse
  return nlkl


import scipy.optimize as opt
def minimize(lkl,pars,fix=None):
  if fix is not None:
    lkl = repar_lkl(lkl,fix)
  chi2 = lkl2chi2(lkl)
  if isinstance(pars,dict):
    p0 = nm.array([pars[p] for p in lkl.paruse])
  else:
    p0 = nm.array(pars)*1.
  return opt.minimize(chi2,p0)

def cut(lkl,trange,fix):
  if len(trange)==2:
    trange = nm.linspace(trange[0],trange[1],100)
  elif len(trange)==3:
    trange = nm.linspace(trange[0],trange[1],trange[2])
  nlkl = repar_lkl(lkl,fix)
  res = nm.array([nlkl([t]) for t in trange])
  return trange,res

class lkl_class:
  def __init__(self,func,paruse):
    self.func = func
    self.paruse = paruse
  def __call__(self,pars):
    if isinstance(pars, dict):
      pars = nm.array([pars[p] for p in paruse])
    return self.func(pars)
  def chi2(self):
    def c2(pars):
      return -2.*self(pars)
    return c2
  def repar(self,fix):
    paruse = [k for k in self.paruse if k not in fix]
    def nlkl(pars):
      if isinstance(pars,dict):
        pdict = dict(zip(paruse,[pars[p] for p in paruse]))
      else:
        pdict = dict(zip(paruse,[p for p in pars]))
      pdict.update(fix)
      return lkl(pdict)
    return lkl_class(nlkl,paruse)

