# STUDENTS: DO NOT CHANGE THIS FILE
from __future__ import division
import numpy as np
import math

# from numpy.compat import integer_types
from six.moves import zip
from tiles import tiles, IHT

# Utility Functions
class Error(Exception):
    pass

# Randomness is only used for reset()-ing the environment
def np_random(seed=None):
    if seed is not None and not (isinstance(seed, int) and 0 <= seed):
        raise Error('Seed must be a non-negative integer or omitted, not {}'.format(seed))
    rng = np.random.RandomState()
    if seed is None:
        seed = rng.randint(2**32 - 1)
    rng.seed(seed)
    return rng, seed

# Environment
class MountainCar:
    def __init__(self, mode=None):
        # Initial positions of box-car
        self.min_position = -1.2
        self.max_position = 0.6
        self.max_speed = 0.07
        self.goal_position = 0.5
        
        self.force=0.001
        self.gravity=0.0025

        # Actions = {0, 1, 2} for go left, do nothing, go right
        self.action_space = 3

        if mode == "tile":
            self.state_space = 2048
        elif mode == 'raw':
            self.state_space = 2
        else:
            raise Error("Invalid environment mode. Must be tile or raw")

        self.mode = mode

        # variables used conditionally on mode or render
        self.iht = None
        self.w = None
        self.viewer = None # needed for render only

        self.seed()
        self.reset()

    def transform(self, state):
        # Normalize values to range from [0, 1] for use in transformations
        position, velocity = state
        position = (position + 1.2) / 1.8
        velocity = (velocity + 0.07) / 0.14
        assert 0 <= position <= 1
        assert 0 <= velocity <= 1
        position *= 2
        velocity *= 2
        if self.mode == "tile":
            if self.iht is None:
                self.iht = IHT(self.state_space)
            tiling = tiles(self.iht, 64, [position, velocity], [0]) + \
                    tiles(self.iht, 64, [position], [1]) + \
                    tiles(self.iht, 64, [velocity], [2])
            # return tiling
            return {index : 1 for index in tiling}
        elif self.mode == "raw":
            return dict(enumerate(state))
        else:
            raise Error("Invalid environment mode. Must be tile or raw")

    def seed(self, seed=None):
        self.np_random, seed = np_random(seed)
        return [seed]

    # def reset(self):
    #     self.state = np.array([self.np_random.uniform(low=-0.6, high=-0.4), 0])
    #     return self.transform(self.state)
    def reset(self):
        self.state = np.array([self.np_random.uniform(low=-0.6, high=-0.4), 0])
        return self.transform(self.state)
    
    def height(self, xs):
        return np.sin(3 * xs)*.45+.55
    
    def step(self, action):
        assert action == 0 or action == 1 or action == 2

        position, velocity = self.state
        velocity += (action-1)*self.force + math.cos(3*position)*(-self.gravity)
        velocity = np.clip(velocity, -self.max_speed, self.max_speed)
        position += velocity
        position = np.clip(position, self.min_position, self.max_position)
        # Left of min_position is a wall
        if (position==self.min_position and velocity<0): velocity = 0
        
        done = bool(position >= self.goal_position)
        reward = -1.0

        self.state = (position, velocity)
        return self.transform(self.state), reward, done

    def render(self, mode='human'):
        # DO NOT TRANSLATE TO OTHER LANGUAGES UNLESS YOU ARE BRAVE
        screen_width = 600
        screen_height = 400

        world_width = self.max_position - self.min_position
        scale = screen_width/world_width
        carwidth=40
        carheight=20

        if self.viewer is None:
            import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            xs = np.linspace(self.min_position, self.max_position, 100)
            ys = self.height(xs)
            xys = list(zip((xs-self.min_position)*scale, ys*scale))

            self.track = rendering.make_polyline(xys)
            self.track.set_linewidth(4)
            self.viewer.add_geom(self.track)

            clearance = 10

            l,r,t,b = -carwidth/2, carwidth/2, carheight, 0
            car = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
            car.add_attr(rendering.Transform(translation=(0, clearance)))
            self.cartrans = rendering.Transform()
            car.add_attr(self.cartrans)
            self.viewer.add_geom(car)
            frontwheel = rendering.make_circle(carheight/2.5)
            frontwheel.set_color(.5, .5, .5)
            frontwheel.add_attr(rendering.Transform(translation=(carwidth/4,clearance)))
            frontwheel.add_attr(self.cartrans)
            self.viewer.add_geom(frontwheel)
            backwheel = rendering.make_circle(carheight/2.5)
            backwheel.add_attr(rendering.Transform(translation=(-carwidth/4,clearance)))
            backwheel.add_attr(self.cartrans)
            backwheel.set_color(.5, .5, .5)
            self.viewer.add_geom(backwheel)
            flagx = (self.goal_position-self.min_position)*scale
            flagy1 = self.height(self.goal_position)*scale
            flagy2 = flagy1 + 50
            flagpole = rendering.Line((flagx, flagy1), (flagx, flagy2))
            self.viewer.add_geom(flagpole)
            flag = rendering.FilledPolygon([(flagx, flagy2), (flagx, flagy2-10), (flagx+25, flagy2-5)])
            flag.set_color(.8,.8,0)
            self.viewer.add_geom(flag)

        pos = self.state[0]
        self.cartrans.set_translation((pos-self.min_position)*scale, self.height(pos)*scale)
        self.cartrans.set_rotation(math.cos(3 * pos))

        return self.viewer.render(return_rgb_array = mode=='rgb_array')
    
    def close(self):
        # DO NOT TRANSLATE TO OTHER LANGUAGES UNLESS YOU ARE BRAVE
        if self.viewer:
            self.viewer.close()
            self.viewer = None
    