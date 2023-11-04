from functools import partial
from astropy.constants import G
import astropy.coordinates as coord
import astropy.units as u
# gala
import gala.coordinates as gc
import gala.dynamics as gd
import gala.potential as gp
from gala.units import dimensionless, galactic, UnitSystem

import jax
import jax.numpy as jnp

from jax.config import config
config.update("jax_enable_x64", True)
import jax.random as random 
from jax_cosmo.scipy.interpolate import InterpolatedUnivariateSpline
usys = UnitSystem(u.kpc, u.Myr, u.Msun, u.radian)

class Potential:
    
    def __init__(self, units, params):
        if units is None:
            units = dimensionless
        self.units = UnitSystem(units)
        
        if self.units == dimensionless:
            self._G = 1
        else:
            self._G = G.decompose(self.units).value
        
        for name, param in params.items():
            if hasattr(param, 'unit'):
                param = param.decompose(self.units).value
            setattr(self, name, param)
    
    @partial(jax.jit, static_argnums=(0,))
    def gradient(self, xyz, t):
        grad_func = jax.grad(self.potential)
        return grad_func(xyz, t)
    
    @partial(jax.jit, static_argnums=(0,))
    def density(self, xyz, t):
        lap = jnp.trace(jax.hessian(self.potential)(xyz, t))
        return lap / (4 * jnp.pi * self._G)
    
    @partial(jax.jit, static_argnums=(0,))
    def acceleration(self, xyz, t):
        return -self.gradient(xyz, t)
    
   
    @partial(jax.jit,static_argnums=(0,))
    def jacobian_force_mw(self, xyz, t):
        jacobian_force_mw = jax.jacfwd(self.gradient)
        return jacobian_force_mw(xyz, t)
    
    @partial(jax.jit,static_argnums=(0,))
    def d2phidr2_mw(self, x, t):
        """
        Computes the second derivative of the Milky Way potential at a position x (in the simulation frame)
        Args:
          x: 3d position (x, y, z) in [kpc]
        Returns:
          Second derivative of force (per unit mass) in [1/Myr^2]
        Examples
        --------
        >>> d2phidr2_mw(x=jnp.array([8.0, 0.0, 0.0]))
        """
        rad = jnp.linalg.norm(x)
        r_hat = x/rad
        dphi_dr_func = lambda x: jnp.sum(self.gradient(x,t)*r_hat)
        return jnp.sum(jax.grad(dphi_dr_func)(x)*r_hat)
        
        ##return jnp.matmul(jnp.transpose(x), jnp.matmul(self.jacobian_force_mw(x, t), x)) / rad**2


    @partial(jax.jit,static_argnums=(0,))
    def omega(self, x,v):
        """
        Computes the magnitude of the angular momentum in the simulation frame
        Args:
          x: 3d position (x, y, z) in [kpc]
          v: 3d velocity (v_x, v_y, v_z) in [kpc/Myr]
        Returns:
          Magnitude of angular momentum in [rad/Myr]
        Examples
        --------
        >>> omega(x=jnp.array([8.0, 0.0, 0.0]), v=jnp.array([8.0, 0.0, 0.0]))
        """
        rad = jnp.sqrt(x[0] ** 2 + x[1] ** 2 + x[2] ** 2)
        omega_vec = jnp.cross(x, v) / (rad**2)
        return jnp.linalg.norm(omega_vec)

    @partial(jax.jit,static_argnums=(0,))
    def tidalr_mw(self, x, v, Msat, t):
        """
        Computes the tidal radius of a cluster in the potential
        Args:
          x: 3d position (x, y, z) in [kpc]
          v: 3d velocity (v_x, v_y, v_z) in [kpc/Myr]
          Msat: Cluster mass in [Msol]
        Returns:
          Tidal radius of the cluster in [kpc]
        Examples
        --------
        >>> tidalr_mw(x=jnp.array([8.0, 0.0, 0.0]), v=jnp.array([8.0, 0.0, 0.0]), Msat=1e4)
        """
        return (self._G * Msat / ( self.omega(x, v) ** 2 - self.d2phidr2_mw(x, t)) ) ** (1.0 / 3.0)
    
    @partial(jax.jit,static_argnums=(0,))
    def lagrange_pts(self,x,v,Msat, t):
        r_tidal = self.tidalr_mw(x,v,Msat, t)
        r_hat = x/jnp.linalg.norm(x)
        L_close = x - r_hat*r_tidal
        L_far = x + r_hat*r_tidal
        return L_close, L_far
    
    @partial(jax.jit,static_argnums=(0,))
    def release_model(self, x, v, Msat,i, t):
        """
        Simplification of particle spray: just release particles in gaussian blob at each lagrange point.
        User sets the spatial and velocity dispersion for the "leaking" of particles
        TODO: change random key handling... need to do all of the sampling up front...
        """
        keya = jax.random.PRNGKey(i*13)
        keyb = jax.random.PRNGKey(i*23)
        
        keyc = jax.random.PRNGKey(i*27)
        keyd = jax.random.PRNGKey(i*3)
        keye = jax.random.PRNGKey(i*17)
        
        L_close, L_far = self.lagrange_pts(x,v,Msat, t) # each is an xyz array
        
        omega_val = self.omega(x,v)
        
        
        r = jnp.linalg.norm(x)
        r_hat = x/r
        r_tidal = self.tidalr_mw(x,v,Msat, t)
        rel_v = omega_val*r_tidal #relative velocity
        
        #circlar_velocity
        dphi_dr = jnp.sum(self.gradient(x, t)*r_hat)
        v_circ = rel_v##jnp.sqrt( r*dphi_dr )
        
        L_vec = jnp.cross(x,v)
        z_hat = L_vec / jnp.linalg.norm(L_vec)
        
        phi_vec = v - jnp.sum(v*r_hat)*r_hat
        phi_hat = phi_vec/jnp.linalg.norm(phi_vec)
        vt_sat = jnp.sum(v*phi_hat)
        
        
        kr_bar = 2.0
        kvphi_bar = 0.3
        ####################kvt_bar = 0.3 ## FROM GALA
        
        kz_bar = 0.0
        kvz_bar = 0.0
        
        sigma_kr = 0.5
        sigma_kvphi = 0.5
        sigma_kz = 0.5
        sigma_kvz = 0.5
        ##############sigma_kvt = 0.5 ##FROM GALA
        
        kr_samp =  kr_bar + jax.random.normal(keya,shape=(1,))*sigma_kr
        kvphi_samp = kr_samp*(kvphi_bar  + jax.random.normal(keyb,shape=(1,))*sigma_kvphi)
        kz_samp = kz_bar + jax.random.normal(keyc,shape=(1,))*sigma_kz
        kvz_samp = kvz_bar + jax.random.normal(keyd,shape=(1,))*sigma_kvz
        ########kvt_samp = kvt_bar + jax.random.normal(keye,shape=(1,))*sigma_kvt
        
        ## Trailing arm
        pos_trail = x + kr_samp*r_hat*(r_tidal) #nudge out
        pos_trail  = pos_trail + z_hat*kz_samp*(r_tidal/1.0)#r #nudge above/below orbital plane
        v_trail = v + (0.0 + kvphi_samp*v_circ*(1.0))*phi_hat#v + (0.0 + kvphi_samp*v_circ*(-r_tidal/r))*phi_hat #nudge velocity along tangential direction
        v_trail = v_trail + (kvz_samp*v_circ*(1.0))*z_hat#v_trail + (kvz_samp*v_circ*(-r_tidal/r))*z_hat #nudge velocity along vertical direction
        
        ## Leading arm
        pos_lead = x + kr_samp*r_hat*(-r_tidal) #nudge in
        pos_lead  = pos_lead + z_hat*kz_samp*(-r_tidal/1.0)#r #nudge above/below orbital plane
        v_lead = v + (0.0 + kvphi_samp*v_circ*(-1.0))*phi_hat#v + (0.0 + kvphi_samp*v_circ*(r_tidal/r))*phi_hat #nudge velocity along tangential direction
        v_lead = v_lead + (kvz_samp*v_circ*(-1.0))*z_hat#v_lead + (kvz_samp*v_circ*(r_tidal/r))*z_hat #nudge velocity against vertical direction
        
    
        
        
        return pos_lead, pos_trail, v_lead, v_trail
    
    @partial(jax.jit,static_argnums=(0,))
    def gen_stream_ics(self, ts, prog_w0, Msat):
        ws_jax = leapfrog_run(prog_w0, ts, self.gradient)
        
        def scan_fun(carry, t):
            i, pos_close, pos_far, vel_close, vel_far = carry
            pos_close_new, pos_far_new, vel_close_new, vel_far_new = self.release_model(ws_jax[i,:3], ws_jax[i,3:], Msat,i, t)
            return [i+1, pos_close_new, pos_far_new, vel_close_new, vel_far_new], [pos_close_new, pos_far_new, vel_close_new, vel_far_new]#[i+1, pos_close_new, pos_far_new, vel_close_new, vel_far_new]
            
            
        #init_carry = [0, 0, 0, 0, 0]
        init_carry = [0, jnp.array([0.0,0.0,0.]), jnp.array([0.0,0.0,0.]), jnp.array([0.0,0.0,0.]), jnp.array([0.0,0.0,0.])] 
        final_state, all_states = jax.lax.scan(scan_fun, init_carry, ts[1:])
        pos_close_arr, pos_far_arr, vel_close_arr, vel_far_arr = all_states
        return pos_close_arr, pos_far_arr, vel_close_arr, vel_far_arr
    
    @partial(jax.jit,static_argnums=(0,))
    def gen_stream(self, ts, prog_w0, Msat):
        pos_close_arr, pos_far_arr, vel_close_arr, vel_far_arr = self.gen_stream_ics(ts, prog_w0, Msat)
        
        def scan_fun(carry, particle_idx):
            i, pos_close_curr, pos_far_curr, vel_close_curr, vel_far_curr = carry
            curr_particle_w0_close = jnp.hstack([pos_close_curr,vel_close_curr])
            curr_particle_w0_far = jnp.hstack([pos_far_curr,vel_far_curr])
            #print(curr_particle_w0_close.shape)
            #indices = jnp.arange(i,len(ts))
            #t_particle = jnp.take(ts, indices)
            #################t_particle = jax.lax.slice(ts,(i,),(len(ts),))
            w_particle_close = leapfrog_run(curr_particle_w0_close, ts, self.gradient)
            w_particle_far = leapfrog_run(curr_particle_w0_far, ts, self.gradient)
            
            ###ts_update = jax.lax.dynamic_slice(ts_curr,(i+1,),(len(ts_curr)-1,))
            
            return [i+1, pos_close_arr[i+1,:], pos_far_arr[i+1,:], vel_close_arr[i+1,:], vel_far_arr[i+1,:]], [w_particle_close, w_particle_far]
        init_carry = [0, pos_close_arr[0,:], pos_far_arr[0,:], vel_close_arr[0,:], vel_far_arr[0,:]]
        particle_ids = jnp.arange(len(pos_close_arr))
        final_state, all_states = jax.lax.scan(scan_fun, init_carry, particle_ids)
        lead_arm, trail_arm = all_states
        return lead_arm, trail_arm
            
    @partial(jax.jit,static_argnums=(0,))
    def gen_stream_final(self, ts, prog_w0, Msat):
        pos_close_arr, pos_far_arr, vel_close_arr, vel_far_arr = self.gen_stream_ics(ts, prog_w0, Msat,)
        
        def scan_fun(carry, particle_idx):
            i, pos_close_curr, pos_far_curr, vel_close_curr, vel_far_curr = carry
            curr_particle_w0_close = jnp.hstack([pos_close_curr,vel_close_curr])
            curr_particle_w0_far = jnp.hstack([pos_far_curr,vel_far_curr])
            #print(curr_particle_w0_close.shape)
            #indices = jnp.arange(i,len(ts))
            #t_particle = jnp.take(ts, indices)
            
            #######t_particle = jax.lax.slice(ts,(i,),(len(ts),))
            minval, maxval =  ts[i],ts[-1]#jnp.min(jax.lax.dynamic_slice(ts,(i,),(len(ts)-i,))), jnp.max(jax.lax.dynamic_slice(ts,(i,),(len(ts)-i,)))
            ###t_particle = ts.at[:i].set( jnp.nan )
            #######################t_particle = jnp.linspace(minval,maxval,len(ts))#jax.lax.dynamic_slice(ts,minval,(len(ts)-i,))##jax.lax.dynamic_slice(ts,(i+1,),(len(ts)-1,))
            t_particle = jax.vmap(ts_func,in_axes=(0,None),)(ts,ts[i])
            ###################################t_particle = get_t_arr(ts,ts[i]) OLD
            ##print(t_particle)
            w_particle_close = leapfrog_run(curr_particle_w0_close, t_particle, self.gradient)
            w_particle_far = leapfrog_run(curr_particle_w0_far, t_particle, self.gradient)
            
            w_particle_close = w_particle_close[-1,:]#w_particle_close[-1,:]
            w_particle_far = w_particle_far[-1,:]#w_particle_far[-1,:]
            
            
            return [i+1, pos_close_arr[i+1,:], pos_far_arr[i+1,:], vel_close_arr[i+1,:], vel_far_arr[i+1,:]], [w_particle_close, w_particle_far]
        init_carry = [0, pos_close_arr[0,:], pos_far_arr[0,:], vel_close_arr[0,:], vel_far_arr[0,:]]
        particle_ids = jnp.arange(len(pos_close_arr))
        #print(len(particle_ids))
        final_state, all_states = jax.lax.scan(scan_fun, init_carry, particle_ids)
        #print(final_state)
        lead_arm, trail_arm = all_states
        return lead_arm, trail_arm##lead_arm, trail_arm
    
    @partial(jax.jit,static_argnums=(0,))
    def gen_stream_vmapped(self, ts, prog_w0, Msat):
        pos_close_arr, pos_far_arr, vel_close_arr, vel_far_arr = self.gen_stream_ics(ts, prog_w0, Msat)
        @jax.jit
        def single_particle_integrate(particle_number,pos_close_curr,pos_far_curr,vel_close_curr,vel_far_curr):
            curr_particle_w0_close = jnp.hstack([pos_close_curr,vel_close_curr])
            curr_particle_w0_far = jnp.hstack([pos_far_curr,vel_far_curr])
            t_particle = jax.vmap(ts_func,in_axes=(0,None),)(ts,ts[particle_number])
            w_particle_close = leapfrog_run(curr_particle_w0_close, t_particle, self.gradient)
            w_particle_far = leapfrog_run(curr_particle_w0_far, t_particle, self.gradient)
            
            w_particle_close = w_particle_close[-1,:]
            w_particle_far = w_particle_far[-1,:]
            
            
            return w_particle_close, w_particle_far
        particle_ids = jnp.arange(len(pos_close_arr))
        
        return jax.vmap(single_particle_integrate,in_axes=(0,0,0,0,0,))(particle_ids,pos_close_arr, pos_far_arr, vel_close_arr, vel_far_arr)

    

class Isochrone(Potential):
    
    def __init__(self, m, a, units=None):
        super().__init__(units, {'m': m, 'a': a})
    
    @partial(jax.jit, static_argnums=(0,))
    def potential(self, xyz, t):
        r = jnp.linalg.norm(xyz, axis=0)
        return - self._G * self.m / (self.a + jnp.sqrt(r**2 + self.a**2))
    

    
class Isochrone_centered(Potential):
    
    def __init__(self, m, a, spline_eval_func, splines, t_min, t_max, m_ext, a_ext, units=None):
        super().__init__(units, {'m': m, 'a': a, 'spline_eval_func': spline_eval_func, 'splines': splines, 't_min': t_min, 't_max': t_max,
                                'm_ext': m_ext, 'a_ext': a_ext})
    
    @partial(jax.jit, static_argnums=(0,))
    def potential(self, xyz, t):
        is_cond_met = (t > self.t_min) & (t < self.t_max) # True if yes, False if no
        pot_ext = Isochrone(m=self.m_ext, a=self.a_ext, units=self.units) 
        
        def true_func(xyz_t):
            xyz_, t = xyz_t[:3], xyz_t[-1]
            xyz = xyz_ - self.spline_eval_func(t,self.splines)
            r = jnp.linalg.norm(xyz, axis=0)
            return - self._G * self.m / (self.a + jnp.sqrt(r**2 + self.a**2))  + pot_ext.potential(xyz_,t)#+ self.pot_ext.potential(xyz_,t)
        def false_func(xyz_t):
            xyz, t = xyz_t[:3], xyz_t[-1]
            return pot_ext.potential(xyz,t)#0.#self.pot_ext.potential(xyz,t)
        xyz_t = jnp.hstack([xyz,t])
        return jax.lax.cond(pred=is_cond_met, true_fun=true_func, false_fun=false_func,operand=xyz_t)
    
class MiyamotoNagaiDisk(Potential):
    def __init__(self, m, a, b, units=None):
        super().__init__(units, {'m': m, 'a': a, 'b': b,})
    @partial(jax.jit,static_argnums=(0,))
    def potential(self,xyz,t):
        R2 = xyz[0]**2 + xyz[1]**2
        return -self._G*self.m / jnp.sqrt(R2 + jnp.square(jnp.sqrt(xyz[2]**2 + self.b**2) + self.a))
    
class NFWPotential_holder(Potential):
    """
    Flattening in potential, not density
    Form from http://gala.adrian.pw/en/v0.1.2/api/gala.potential.FlattenedNFWPotential.html
    """
    def __init__(self, v_c, r_s, q, units=None):
        super().__init__(units, {'v_c': v_c, 'r_s': r_s, 'q': q})
    @partial(jax.jit,static_argnums=(0,))
    def potential(self,xyz,t):
        m = jnp.sqrt(xyz[0]**2 + xyz[1]**2 + (xyz[2]/self.q)**2)
        return -((self.v_c**2)/jnp.sqrt(jnp.log(2.0)-0.5) )*jnp.log(1.0 + m/self.r_s)/(m/self.r_s)
    
class NFWPotential(Potential):
    """
    standard def see spherical model @ https://github.com/adrn/gala/blob/main/gala/potential/potential/builtin/builtin_potentials.c
    """
    def __init__(self, m, r_s, units=None):
        super().__init__(units, {'m': m, 'r_s': r_s})
    @partial(jax.jit,static_argnums=(0,))
    def potential(self,xyz,t):
        v_h2 = -self._G*self.m/self.r_s
        m = jnp.sqrt(xyz[0]**2 + xyz[1]**2 + xyz[2]**2 + .001)/self.r_s ##added softening!
        return v_h2*jnp.log(1.0+ m) / m#-((self.v_c**2)/jnp.sqrt(jnp.log(2.0)-0.5) )*jnp.log(1.0 + m/self.r_s)/(m/self.r_s)
    
    
class BarPotential(Potential):
    """
    Rotating bar potentil, with hard-coded rotation.
    Eq 8a in https://articles.adsabs.harvard.edu/pdf/1992ApJ...397...44L
    Rz according to https://en.wikipedia.org/wiki/Rotation_matrix
    """
    def __init__(self, m, a, b, c, Omega, units=None):
        super().__init__(units, {'m': m, 'a': a, 'b': b, 'c': c, 'Omega': Omega})
    @partial(jax.jit,static_argnums=(0,))
    def potential(self,xyz,t):
        ## First take the simulation frame coordinates and rotate them by Omega*t
        ang = -self.Omega*t
        Rot_mat = jnp.array([[jnp.cos(ang), -jnp.sin(ang), 0], [jnp.sin(ang), jnp.cos(ang), 0.], [0.0, 0.0, 1.0] ])
        Rot_inv = jnp.linalg.inv(Rot_mat)
        xyz_corot = jnp.matmul(Rot_mat,xyz)
        
        T_plus = jnp.sqrt( (self.a + xyz_corot[0])**2 + xyz_corot[1]**2 + ( self.b + jnp.sqrt(self.c**2 + xyz_corot[2]**2) )**2 )
        T_minus = jnp.sqrt( (self.a - xyz_corot[0])**2 + xyz_corot[1]**2 + ( self.b + jnp.sqrt(self.c**2 + xyz_corot[2]**2) )**2 )
        
        pot_corot_frame = (self._G*self.m/(2.0*self.a))*jnp.log( (xyz_corot[0] - self.a + T_minus)/(xyz_corot[0] + self.a + T_plus) )
        return pot_corot_frame
    
    
class Potential_Combine(Potential):
    def __init__(self, potential_list, units=None):
        super().__init__(units, {'potential_list': potential_list })
    
    @partial(jax.jit,static_argnums=(0,))
    def potential(self, xyz, t,):
        output = []
        for i in range(len(self.potential_list)):
            output.append(self.potential_list[i].potential(xyz,t))
        return jnp.sum(jnp.array(output))
    
    
def leapfrog_step(func, y0, t0, dt, a0):
    ndim = y0.shape[0] // 2
    tf = t0 + dt
    
    x0 = y0[:ndim]
    v0 = y0[ndim:]
    
    v1_2 = v0 + a0 * dt / 2.
    xf = x0 + v1_2 * dt
    af = - func(xf, tf)
    
    vf = v1_2 + af * dt / 2
    
    return tf, jnp.concatenate((xf, vf)), af

@partial(jax.jit, static_argnames=['potential_gradient', 'args'])
def leapfrog_run(w0, ts, potential_gradient, args=()):
    func_ = lambda y, t: potential_gradient(y, t, *args)
    
    def scan_fun(carry, t):
        i, y0, t0, dt, a0 = carry
        tf, yf, af = leapfrog_step(func_, y0, t0, dt, a0)
        dt_new = ts[i+1] - ts[i]
        is_cond_met = dt_new > 0.
        
        def true_func(dt_new):
            return ts[-1]-ts[-2] #dt_base !!!ASSUMING dt = 0.5 Myr by default!!!!
        def false_func(dt_new):
            return 0.0
        dt_new = jax.lax.cond(pred=is_cond_met, true_fun=true_func, false_fun=false_func,operand=dt_new)
        
        ###tf = tf + dt_new
        return [i + 1, yf, tf, dt_new, af], yf
    
    ndim = w0.shape[0] // 2
    a0 = func_(w0[:ndim], ts[0])
    dt = ts[1]-ts[0] ## I ADDED THIS
    init_carry = [0, w0, ts[0], dt, a0]
    _, ws = jax.lax.scan(scan_fun, init_carry, ts[1:])
    res_ws = jnp.concatenate((w0[None], ws))
    
    return res_ws

def compute_orbit_and_gradients(params,w0,dt,n_steps,units):
    pot = Hernquist(m=params['m'], a=params['a'],units=units,)  # Create the potential with the given parameters
    
    # Example Orbit:
    #w0 = jnp.array([10, 0, 0, 0, .204, 0])
    t0 = 0
    #dt = 0.1
    #n_steps = 10000
    ts = jnp.arange(t0, t0 + dt * n_steps + dt/10., dt)
    
    ### Compute the orbit with the provided potential
    ws_jax = leapfrog_run(w0, ts, pot.gradient)
    
    # Compute the gradient of the final position with respect to the potential parameters
    def gradient_fn(params):
        pot_with_params = Hernquist(m=params['m'], a=params['a'],units=units)
        ws_with_params = leapfrog_run(w0, ts, pot_with_params.gradient)
        final_position = ws_with_params[-1, :]  # Extract final position
        return final_position
    
    # Compute the gradient using JAX's autodiff
    gradient = jax.jacfwd(gradient_fn)(params)
    
    return ws_jax, gradient

def get_stripping_time_arr(ts):
    """
    assuming a particle is released for index in the ts array
    TODO: Custom release
    """
    t_release_arr = []
    for i in range(len(ts)):
        t_release_arr.append( ts[i:] )
    return t_release_arr

#def get_spl_funcs(ts,ws):
#    return [InterpolatedUnivariateSpline(ts,ws[:,i],k=3) for i in range(3)]

#def eval_spl_funcs(ts, spl_funcs):
#    return jnp.array([spl_funcs[i](ts) for i in range(3)])

@jax.jit
def get_t_arr(ts,t_min):
    def scan_fun(carry, t):
        i, ts_curr = carry
        
        def true_func(ts_curr):
            return ts_curr
        def false_func(ts_curr):
            return 0.0
        is_cond_met = ts_curr > t_min
        val_fill = jax.lax.cond(pred=is_cond_met, true_fun=true_func, false_fun=false_func,operand=ts_curr)
        
        return [i + 1,ts[i+1]], val_fill
    init_carry = [0,ts[0]]
    __, t_particle = jax.lax.scan(scan_fun, init_carry, ts)
    return t_particle

@jax.jit
def ts_func(t,t_min):
    def true_func(ts_curr):
        return ts_curr
    def false_func(ts_curr):
        return 0.0
    is_cond_met = t > t_min
    return jax.lax.cond(pred=is_cond_met, true_fun=true_func, false_fun=false_func,operand=t)



@jax.jit
def get_rot_mat(n_hat):
    """
    Get rotation matrix that transforms from INERTIAL coordinates to SATELLITE coordinates
    https://math.stackexchange.com/questions/1956699/getting-a-transformation-matrix-from-a-normal-vector
    """
    nx,ny,nz = n_hat
    nx_ny = jnp.sqrt(nx**2 + ny**2)
    return jnp.array([[ny/nx_ny, -nx/nx_ny, 0], [nx*nz/nx_ny, ny*nz/nx_ny, -nx_ny], [nx,ny,nz]])

def get_spl_funcs(ts,ws):
    return [InterpolatedUnivariateSpline(ts,ws[:,0],k=3),InterpolatedUnivariateSpline(ts,ws[:,1],k=3), InterpolatedUnivariateSpline(ts,ws[:,2],k=3)]

def eval_spl_funcs(ts, spl_funcs):
    return jnp.array([spl_funcs[0](ts),spl_funcs[1](ts),spl_funcs[2](ts)])