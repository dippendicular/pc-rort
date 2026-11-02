"""Observable PC-RORT estimator for the audited exact regular model.

Known loadings and curvature; independent encoder, private selector and fresh
fit partitions.  Inference is unconditional asymptotic in fixed free
coordinates, not a finite-sample or neural-network coverage certificate.
"""
from __future__ import annotations
from dataclasses import dataclass
import hashlib
import numpy as np
from scipy.stats import chi2
from scipy.optimize import minimize
from scipy.special import expit
from .matrix import conditional_risk_matrix
from .method_revision import empirical_bernstein_gate


class UnsupportedInference(ValueError):
    """The declared interior full-rank inference path is unavailable."""


def svec(a):
    a=np.asarray(a,float);ii=np.triu_indices(len(a));v=a[ii].copy()
    v[ii[0]!=ii[1]]*=np.sqrt(2)
    return v


def smat(v,p):
    ii=np.triu_indices(p);v=np.asarray(v,float).copy()
    if v.shape!=(p*(p+1)//2,):raise ValueError('invalid svec dimension')
    v[ii[0]!=ii[1]]/=np.sqrt(2)
    a=np.zeros((p,p));a[ii]=v;a[(ii[1],ii[0])]=v
    return a


def free_pack(b,sigma):
    """b is p by J; sigma comprises J scaled private p by p blocks."""
    return np.r_[np.asarray(b).T.ravel(),*(svec(c) for c in sigma)]


def free_unpack(gamma,j,p):
    gamma=np.asarray(gamma,float);n=j*p;s=p*(p+1)//2
    if gamma.shape!=(n+j*s,):raise ValueError('invalid free dimension')
    return gamma[:n].reshape(j,p).T,np.stack([smat(gamma[n+k*s:n+(k+1)*s],p) for k in range(j)])


def joint_blocks(private):
    j,p,_=np.asarray(private).shape;out=np.zeros((j,p,j,p))
    for k,c in enumerate(private):out[k,:,k,:]=c
    return out


def risk_components(b,h,conditional_covariance):
    """Return B, V and Q; covariance-only ablation preserves B exactly."""
    q=conditional_risk_matrix(b,h,conditional_covariance)
    bias=np.asarray(b).T@h@np.asarray(b)
    v=q-bias
    return bias,v,q


def covariance_only_ablation(b,h,conditional_covariance):
    bias,v,_=risk_components(b,h,conditional_covariance)
    return bias+np.diag(np.diag(v))


@dataclass(frozen=True)
class WaldImage:
    factor: np.ndarray
    critical: float
    radius: float
    linear_radius: float
    quadratic_radius: float
    arithmetic_guard: float
    min_studentizer_eigenvalue: float
    status: str='analytic-enclosure-evaluated-in-floating-point'
    verified_numerical_certificate: bool=False


@dataclass(frozen=True)
class RegularEstimate:
    encoder_mean: np.ndarray
    pilot_operators: np.ndarray
    bias: np.ndarray
    curvature: np.ndarray
    scaled_private_covariance: np.ndarray
    conditional_covariance: np.ndarray
    gamma: np.ndarray
    influence_covariance: np.ndarray
    influence_groups: tuple
    q: np.ndarray
    normalization: int
    final_sizes: np.ndarray
    declared_interior: bool

    def image(self,alpha=.05):
        if not 0<alpha<1:raise ValueError('alpha must be in (0,1)')
        if not self.declared_interior:raise UnsupportedInference('PSD boundary or undeclared interior model: LOCAL fallback')
        psi=self.influence_covariance;d=len(self.gamma);n=self.normalization
        try:chol=np.linalg.cholesky(psi)
        except np.linalg.LinAlgError as exc:raise UnsupportedInference('full-rank studentization unavailable; no pseudoinverse truncation') from exc
        mineig=float(np.linalg.eigvalsh(psi).min())
        if mineig<=0:raise UnsupportedInference('studentizer not positive definite')
        if any(np.linalg.eigvalsh(c).min()<=0 for c in self.scaled_private_covariance):
            raise UnsupportedInference('private covariance center not interior')
        critical=float(chi2.ppf(1-alpha,d));a=np.sqrt(critical/n)*chol
        p,j=self.bias.shape
        jac=[]
        for col in a.T:
            x,t=free_unpack(col,j,p)
            dq=x.T@self.curvature@self.bias+self.bias.T@self.curvature@x
            dq+=np.diag([np.trace(self.curvature@c)/n for c in t])
            jac.append(dq.ravel())
        linear=float(np.linalg.norm(np.column_stack(jac),'fro'))
        ab=float(np.linalg.norm(a[:j*p],'fro'))
        quadratic=float(np.linalg.norm(self.curvature,2)*ab**2)
        guard=float(128*np.finfo(float).eps*(1+np.linalg.norm(self.q,'fro')+linear+quadratic))
        return WaldImage(a,critical,linear+quadratic+guard,linear,quadratic,guard,mineig)

    def target_statistic(self,target_gamma):
        """Oracle-only validation helper; never used by selection."""
        error=self.gamma-np.asarray(target_gamma,float)
        return float(self.normalization*error@np.linalg.solve(self.influence_covariance,error))


def estimate_regular(encoder,selectors,loadings,curvature,final_sizes,*,interior_model=False):
    """Estimate exact conditional moments using pilot information only.

    The encoder and selector rows are independent clusters. Loadings have shape
    (J,p,d_encoder), with a full-row-rank local loading in this implemented
    regime. Fixed H and structurally zero off-site conditional covariance
    coordinates are omitted, not estimated and truncated.
    """
    w=np.asarray(encoder,float);xs=tuple(np.asarray(x,float) for x in selectors)
    loads=np.asarray(loadings,float);h=np.asarray(curvature,float);nf=np.asarray(final_sizes,int)
    if w.ndim!=2 or len(w)<3 or not xs:raise ValueError('invalid encoder or selector partitions')
    j=len(xs);p=xs[0].shape[1]
    if loads.shape!=(j,p,w.shape[1]) or h.shape!=(p,p) or nf.shape!=(j,):raise ValueError('incompatible shapes')
    if not all(x.ndim==2 and x.shape[1]==p and len(x)>p+2 for x in xs):raise ValueError('insufficient selector observations')
    if min(nf)<1 or not all(np.isfinite(x).all() for x in (w,loads,h,*xs)):raise ValueError('nonfinite data or invalid final sizes')
    if not np.allclose(h,h.T) or np.linalg.eigvalsh(h).min()<=0:raise UnsupportedInference('known curvature must be positive definite in the implemented regime')
    if np.linalg.matrix_rank(loads[0])<p:raise UnsupportedInference('local loading requires full row rank; no empirical rank deletion')
    n=len(w);phi=w.mean(0);means=np.stack([x.mean(0) for x in xs])
    operators=means+np.einsum('jpd,d->jp',loads,phi)
    b=(operators-operators[0]).T
    covariance=np.stack([np.atleast_2d(np.cov(x,rowvar=False,ddof=1)) for x in xs])
    scaled=covariance*n/nf[:,None,None]
    omega=joint_blocks(covariance/nf[:,None,None])
    gamma=free_pack(b,scaled);dim=len(gamma);groups=[]
    # Encoder target-error influence: the same local loading in every column.
    enc=np.zeros((n,dim));common=-(w-phi)@loads[0].T
    enc[:,:j*p]=np.tile(common,(1,j));groups.append(enc)
    size=p*(p+1)//2
    for site,x in enumerate(xs):
        res=x-x.mean(0);kappa=len(x)/n;kfit=nf[site]/n
        values=np.zeros((len(x),dim))
        for candidate in range(j):
            coeff=int(site==candidate)-int(site==0)
            values[:,candidate*p:(candidate+1)*p]=coeff*res/kappa
        outer=np.einsum('ni,nj->nij',res,res);center=outer.mean(0)
        covif=np.stack([svec(a-center) for a in outer])/(kappa*kfit)
        start=j*p+site*size;values[:,start:start+size]=covif
        groups.append(values)
    psi=sum(g.T@g for g in groups)/n
    q=conditional_risk_matrix(b,h,omega)
    arrays=[phi,operators,b,h,scaled,omega,gamma,psi,nf]
    for array in arrays:array.setflags(write=False)
    for g in groups:g.setflags(write=False)
    return RegularEstimate(phi,operators,b,h,scaled,omega,gamma,psi,tuple(groups),q,n,nf,bool(interior_model))


def center_weights(effective,lam):
    v=np.asarray(effective,float)
    if lam<0 or not np.isfinite(lam):raise ValueError('invalid lambda')
    rho=lam/(1+lam)
    if rho==0:
        if not np.allclose(v,np.eye(1,len(v),0)[0]):raise ValueError('lambda zero requires LOCAL')
        return v.copy()
    if min(v)<-1e-8 or abs(v.sum()-1)>1e-8 or v[0]<1-rho-1e-8:raise ValueError('effective weights violate simplex or mass cap')
    out=v/rho;out[0]=(v[0]-1+rho)/rho
    return out


class LockedSelection:
    """One-way lock/release contract; no final observations before lock."""
    def __init__(self,estimate,lam):
        self.estimate=estimate;self.lam=float(lam);self.weights=None;self.events=[];self.final=None

    def lock(self,v):
        if self.weights is not None:raise RuntimeError('selection already locked')
        v=np.array(v,float,copy=True);center_weights(v,self.lam)
        v.setflags(write=False);self.weights=v
        self.digest=hashlib.sha256(v.tobytes()).hexdigest();self.events.append('weights_locked')

    def release(self,fresh_private_samples,loadings):
        if self.weights is None:raise RuntimeError('final release prohibited before weight lock')
        if self.final is not None:raise RuntimeError('final release already consumed')
        if len(fresh_private_samples)!=len(self.weights):raise ValueError('wrong final candidate count')
        if any(len(x)!=n for x,n in zip(fresh_private_samples,self.estimate.final_sizes)):raise ValueError('final sizes differ from pilot inference plan')
        means=np.stack([np.asarray(x,float).mean(0) for x in fresh_private_samples])
        self.final=means+np.einsum('jpd,d->jp',np.asarray(loadings),self.estimate.encoder_mean)
        self.final.setflags(write=False);self.events.append('fresh_final_released')
        return self.final

    def centers(self):
        if self.final is None:raise RuntimeError('final operators unavailable')
        w=center_weights(self.weights,self.lam)
        return self.final[0].copy(),w@self.final


def matched_brier_adaptation(local_center,transported_center,features,outcomes,h,lam):
    """Both predictors solve Eq. (28) using the same bounded Brier loss.

    Numerical optimizer residuals are returned, not an exact-solution claim.
    With the validation design's Rademacher features, H=I/8 and lambda=4p,
    the penalty dominates the global lower curvature bound of Brier loss.
    """
    x=np.asarray(features,float);y=np.asarray(outcomes,float);h=np.asarray(h,float)
    def fit(center):
        def fg(theta):
            p=expit(x@theta);r=p-y;d=theta-center
            return float(np.mean(r*r)+lam/2*d@h@d),2*x.T@(r*p*(1-p))/len(y)+lam*h@d
        result=minimize(fg,np.array(center,float),jac=True,method='BFGS',options={'gtol':1e-9,'maxiter':1000})
        residual=float(np.linalg.norm(fg(result.x)[1]))
        if residual>1e-6:raise RuntimeError('adaptation gradient residual too large')
        return result.x,residual
    local,lr=fit(np.asarray(local_center,float));trans,tr=fit(np.asarray(transported_center,float))
    return local,trans,lr,tr


def paired_brier_gate(local_prediction,transported_prediction,outcomes,delta=.05):
    if not 0<delta<1:raise ValueError('delta must lie in (0,1)')
    lp=np.asarray(local_prediction,float);tp=np.asarray(transported_prediction,float);y=np.asarray(outcomes,float)
    if lp.shape!=tp.shape or lp.shape!=y.shape:raise ValueError('paired calibration shapes differ')
    if not np.isfinite(np.r_[lp,tp,y]).all() or min(lp.min(),tp.min(),y.min())<0 or max(lp.max(),tp.max(),y.max())>1:raise ValueError('Brier inputs must lie in [0,1]')
    mean,upper=empirical_bernstein_gate((tp-y)**2-(lp-y)**2,delta,1.)
    return {'mean':mean,'upper':upper,'accepted':bool(upper<0),'formula':'manuscript Eq. (29)','n':len(y)}
