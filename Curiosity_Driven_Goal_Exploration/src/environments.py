from explauto.environment.environment import Environment


class ExplautoEnv(Environment):
    def __init__(self, m_mins, m_maxs, s_mins, s_maxs):
        Environment.__init__(self, m_mins, m_maxs, s_mins, s_maxs)

        self.reset()

    def reset(self):
        self.poses = None
        self.logs = []

    def compute_motor_command(self, m):
        return m

    def compute_sensori_effect(self, s):
        assert 1 == 2
        self.poses = s
        self.logs.append(self.poses)

        return list(self.poses)
