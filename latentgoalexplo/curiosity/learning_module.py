import numpy as np

from numpy import array, hstack

from explauto.agent import Agent
from explauto.utils import rand_bounds
from explauto.utils.config import make_configuration
from explauto.exceptions import ExplautoBootstrapError
from explauto.sensorimotor_model.non_parametric import NonParametric
from explauto.interest_model.competences import competence_dist

from latentgoalexplo.curiosity.interest_model import MiscRandomInterest, MiscGaussianInterest, MiscDiscretizedInterest


class LearningModule(Agent):
    def __init__(self, mid, m_space, s_space, env_conf, explo_noise, win_size, interest_model):

        explo_noise = explo_noise

        self.conf = make_configuration(env_conf.m_mins[m_space],
                                       env_conf.m_maxs[m_space],
                                       array(list(env_conf.m_mins[m_space]) + list(env_conf.s_mins))[s_space],
                                       array(list(env_conf.m_maxs[m_space]) + list(env_conf.s_maxs))[s_space])

        self.im_dims = self.conf.s_dims

        self.mid = mid
        self.m_space = m_space
        self.s_space = s_space
        self.motor_babbling_n_iter = 10

        self.s = None
        self.last_interest = 0

        if interest_model == 'uniform':
            im_cls, kwargs = (MiscRandomInterest, {
                'competence_measure': competence_dist,
                'win_size': win_size,
                'competence_mode': 'knn',
                'k': 20,
                'progress_mode': 'local'})
        elif interest_model == 'normal':
            im_cls, kwargs = (MiscGaussianInterest, {
                'competence_measure': competence_dist,
                'win_size': win_size,
                'competence_mode': 'knn',
                'k': 20,
                'progress_mode': 'local'})
        elif interest_model == 'active':
            im_cls, kwargs = (MiscDiscretizedInterest, {
                'x_card': 20 ** len(self.im_dims),  # 20 is the number of cells on each dimension
                'cells_win_size': 20,  # window size parameter (ws)
                'eps_random': 0.1,  # proportion of random choice of cell
                'measure': competence_dist,
                'competence_measure': competence_dist,
                'win_size': win_size,
                'competence_mode': 'knn',
                'k': 20,
                'progress_mode': 'local'})
        else:
            raise NotImplementedError

        self.im = im_cls(self.conf, self.im_dims, **kwargs)

        sm_cls, kwargs = (NonParametric, {'fwd': 'NN', 'inv': 'NN', 'sigma_explo_ratio': explo_noise})
        self.sm = sm_cls(self.conf, **kwargs)

        Agent.__init__(self, self.conf, self.sm, self.im)

    def motor_babbling(self, n=1):
        if n == 1:
            return rand_bounds(self.conf.m_bounds)[0]
        else:
            return rand_bounds(self.conf.m_bounds, n)

    def goal_babbling(self):
        s = rand_bounds(self.conf.s_bounds)[0]
        m = self.sm.infer(self.conf.s_dims, self.conf.m_dims, s)
        return m

    def get_m(self, ms):
        return array(ms[self.m_space])

    def get_m_batch(self, ms):
        return array(ms[:, self.m_space])

    def get_s(self, ms):
        return array(ms[self.s_space])

    def get_s_batch(self, ms):
        return array(ms[:, self.s_space])

    def set_one_m(self, ms, m):
        """ Set motor dimensions used by module
        """
        ms = array(ms)
        ms[self.mconf['m']] = m

    def set_m(self, ms, m):
        """ Set motor dimensions used by module on one ms
        """
        self.set_one_m(ms, m)
        if self.mconf['operator'] == "seq":
            return [array(ms), array(ms)]
        elif self.mconf['operator'] == "par":
            return ms
        else:
            raise NotImplementedError

    def set_s(self, ms, s):
        """ Set sensory dimensions used by module
        """
        ms = array(ms)
        ms[self.mconf['s']] = s
        return ms

    def inverse(self, s):
        m = self.infer(self.conf.s_dims, self.conf.m_dims, s, pref='')
        return self.motor_primitive(m)

    def infer(self, expl_dims, inf_dims, x, pref='', n=1, explore=True):
        try:
            if self.n_bootstrap > 0:
                self.n_bootstrap -= 1
                raise ExplautoBootstrapError
            mode = "explore" if explore else "exploit"
            if n == 1:
                self.sensorimotor_model.mode = mode
                m = self.sensorimotor_model.infer(expl_dims, inf_dims, x.flatten())
            else:
                self.sensorimotor_model.mode = mode
                m = []
                for _ in range(n):
                    m.append(self.sensorimotor_model.infer(expl_dims, inf_dims, x.flatten()))
            # self.emit(pref + 'inference' + '_' + self.mid, m)
        except ExplautoBootstrapError:
            if n == 1:
                m = rand_bounds(self.conf.bounds[:, inf_dims]).flatten()
            else:
                m = rand_bounds(self.conf.bounds[:, inf_dims], n)
        return m

    def produce(self, n=1):
        if self.t < self.motor_babbling_n_iter:
            self.m = self.motor_babbling(n)
            self.s = np.zeros(len(self.s_space))
            self.x = np.zeros(len(self.expl_dims))
        else:
            self.x = self.choose()
            self.y = self.infer(self.expl_dims, self.inf_dims, self.x, n=n)
            # self.m, self.s = self.extract_ms(self.x, self.y)
            self.m, sg = self.y, self.x  # self.extract_ms(self.x, self.y)
            # self.m = self.motor_primitive(self.m)

            self.s = sg
            # self.emit('movement' + '_' + self.mid, self.m)
        return self.m

    def update_sm(self, m, s):
        self.sensorimotor_model.update(m, s)
        self.t += 1

    def update_sm_batch(self, m, s):
        self.sensorimotor_model.update_batch(m, s)
        self.t += 1

    def update_im(self, m, s):
        if self.t >= self.motor_babbling_n_iter:
            return self.interest_model.update(hstack((m, self.s)), hstack((m, s)))

    def competence(self):
        return self.interest_model.competence()

    def interest(self):
        return self.interest_model.interest()

    def perceive(self, m, s, has_control=True):
        self.update_sm(m, s)
        if has_control:
            self.last_interest = self.update_im(m, s)
