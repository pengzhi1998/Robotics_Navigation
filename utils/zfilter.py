import numpy as np

# from https://github.com/joschu/modular_rl
# http://www.johndcook.com/blog/standard_deviation/


class RunningStat(object):
    def __init__(self, shape_img_depth, shape_goal, shape_ray):
        self._n = 0
        self._M_img_depth = np.zeros(shape_img_depth)
        self._S_img_depth = np.zeros(shape_img_depth)
        self._M_goal = np.zeros(shape_goal)
        self._S_goal = np.zeros(shape_goal)
        self._M_ray = np.zeros(shape_ray)
        self._S_ray = np.zeros(shape_ray)

    def push(self, img_depth, goal, ray):
        img_depth = np.asarray(img_depth)
        goal = np.asarray(goal)
        ray = np.asarray(ray)
        assert img_depth.shape == self._M_img_depth.shape and\
               goal.shape == self._M_goal.shape and\
            ray.shape == self._M_ray.shape
        self._n += 1
        if self._n == 1:
            self._M_img_depth[...] = img_depth
            self._M_goal[...] = goal
            self._M_ray[...] = ray
        else:
            oldM_img_depth = self._M_img_depth.copy()
            oldM_goal = self._M_goal.copy()
            oldM_ray = self._M_ray.copy()
            self._M_img_depth[...] = oldM_img_depth + (img_depth - oldM_img_depth) / self._n
            self._S_img_depth[...] = self._S_img_depth + (img_depth - oldM_img_depth) * (img_depth - self._M_img_depth)
            self._M_goal[...] = oldM_goal + (goal - oldM_goal) / self._n
            self._S_goal[...] = self._S_goal + (goal - oldM_goal) * (goal - self._M_goal)
            self._M_ray[...] = oldM_ray + (ray - oldM_ray) / self._n
            self._S_ray[...] = self._S_ray + (ray - oldM_ray) * (ray - self._M_ray)

    @property
    def n(self):
        return self._n

    @property
    def mean_img_depth(self):
        return self._M_img_depth

    @property
    def mean_goal(self):
        return self._M_goal

    @property
    def mean_ray(self):
        return self._M_ray

    @property
    def var_img_depth(self):
        return self._S_img_depth / (self._n - 1) if self._n > 1 else np.square(self._M_img_depth)

    @property
    def var_goal(self):
        return self._S_goal / (self._n - 1) if self._n > 1 else np.square(self._M_goal)

    @property
    def var_ray(self):
        return self._S_ray / (self._n - 1) if self._n > 1 else np.square(self._M_ray)

    @property
    def std_img_depth(self):
        return np.sqrt(self.var_img_depth)

    @property
    def std_goal(self):
        return np.sqrt(self.var_goal)

    @property
    def std_ray(self):
        return np.sqrt(self.var_ray)

    @property
    def shape_img_depth(self):
        return self._M_img_depth.shape

    @property
    def shape_goal(self):
        return self._M_goal.shape

    @property
    def shape_ray(self):
        return self._M_ray.shape

class ZFilter:
    """
    y = (x-mean)/std
    using running estimates of mean,std
    """

    def __init__(self, shape_img_depth, shape_goal, shape_ray, demean=True, destd=True, clip=10.0):
        self.demean = demean
        self.destd = destd
        self.clip = clip

        self.rs = RunningStat(shape_img_depth, shape_goal, shape_ray)
        self.fix = False

    def __call__(self, img_depth, goal, ray, update=True):
        if update and not self.fix:
            self.rs.push(img_depth, goal, ray)
        if self.demean:
            img_depth = img_depth - self.rs.mean_img_depth
            goal = goal - self.rs.mean_goal
            ray = ray - self.rs.mean_ray
        if self.destd:
            img_depth = img_depth / (self.rs.std_img_depth + 1e-8)
            goal = goal / (self.rs.std_goal + 1e-8)
            ray = ray / (self.rs.std_ray + 1e-8)
        if self.clip:
            img_depth = np.clip(img_depth, -self.clip, self.clip)
            goal = np.clip(goal, -self.clip, self.clip)
            ray = np.clip(ray, -self.clip, self.clip)
        return img_depth, goal, ray

