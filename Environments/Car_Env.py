import sys
sys.path.append('/home/matthew/Documents/Reinforcement_Learning/')

import math
import time
import numpy as np
import numpy.random as rd

from Resources import Geometry
import torch
import torch.nn as nn

import pyglet
from pyglet.gl import *
from pyglet.window import key

class Discrete:
    def __init__(self, size):
        self.n = size

class MainEnv:
    """ The environment of the race track
    """
    def __init__(self, rand_start = False ):
        self.viewer = None
        self.rand_start = rand_start
        self.action_space = Discrete(12)

        ## The constants defining physics of the model
        self.time       = 0
        self.dt         = 0.2
        self.time_limit = 1000
        self.mass       = 1.0
        self.wheel_base = 3.0
        self.length     = 5.0
        self.width      = 3.0

        self.turn_max   = 0.4
        self.engine_max = 1.0
        self.brake_max  = 0.3

        self.fric_coef   = 0.02
        self.drag_coef   = 0.01

        self.slip_speed = 11
        self.tracs      = 0.3
        self.tracf      = 0.0001
        self.brake_mult = 10

        self.max_ray_lenths = 60
        self.n_fwd_rays = 9
        self.fwd_ray_angles = np.pi/2*np.linspace(-1.1, 1.1, self.n_fwd_rays)

        self.fwd_state   = 0
        self.brk_state   = 0
        self.trn_state   = 0

        ## The list of possible starts for a random start method
        self.possible_starts = [ ( [ 28.0,  54.0  ], [  0.0,  1.0  ], 0  ),
                                 ( [ 70.0,  172.0 ], [  1.0,  0.0  ], 8  ),
                                 ( [ 172.0, 166.0 ], [  0.0, -1.0  ], 13 ),
                                 ( [ 130.0, 140.0 ], [ -1.0,  0.0  ], 16 ),
                                 ( [ 130.0, 116.0 ], [  1.0,  0.0  ], 24 ),
                                 ( [ 176.0, 80.0  ], [  0.0, -1.0  ], 28 ),
                                 ( [ 112.0, 30.0  ], [ -1.0,  0.0  ], 33 ) ]

        self.set_up_track()
        self.reset()

    @property
    def state(self):
        """ State returns a list of the various observables that will be used as inputs for the neural network
            These are the velocity projected along the heading axis
            The forward ray lengths
        """
        raw = np.concatenate(( [self.fwd_vel], [self.sid_vel], self.fwd_ray_lens / self.max_ray_lenths ))
        return raw


    def reset(self):
        """ Resetting to the central vertical position with a tiny velocity so it doesnt stall
        """
        if self.rand_start:
            sel_start = rd.randint( len(self.possible_starts) )
        else:
            sel_start = 0

        self.position       = np.array( self.possible_starts[sel_start][0] )
        self.heading        = np.array( self.possible_starts[sel_start][1] )
        self.n_gates_passed = self.possible_starts[sel_start][2]

        self.velocity = 0.1 * self.heading
        self.time = 0

        self.update_car_vectors()
        self.shoot_rays()

        return self.state


    def set_up_track(self):
        """Creating the track from a list of locations, the width of the playing world is 100
        """

        self.outer_track = 2*np.array( [
                [ 10,  14 ],
                [ 10,  80 ],
                [ 20,  90 ],
                [ 90,  90 ],
                [ 90,  66 ],
                [ 42,  66 ],
                [ 42,  62 ],
                [ 90,  62 ],
                [ 90,  14 ],
                [ 82,  10 ],
                [ 66,  18 ],
                [ 50,  10 ],
                [ 34,  18 ],
                [ 18,  10 ],
        ] )

        self.inner_track = 2*np.array( [
                [ 18,  18 ],
                [ 18,  76 ],
                [ 24,  82 ],
                [ 82,  82 ],
                [ 82,  74 ],
                [ 45,  74 ],
                [ 35,  79 ],
                [ 25,  69 ],
                [ 25,  59 ],
                [ 35,  49 ],
                [ 45,  54 ],
                [ 82,  54 ],
                [ 86,  47 ],
                [ 86,  25 ],
                [ 82,  18 ],
                [ 66,  26 ],
                [ 50,  18 ],
                [ 34,  26 ],
        ] )

        self.reward_gates = 2*np.array([
            [ [10, 30], [18, 30] ],
            [ [10, 40], [18, 40] ],
            [ [10, 50], [18, 50] ],
            [ [10, 60], [18, 60] ],
            [ [10, 70], [18, 70] ],
            [ [10, 80], [18, 76] ],
            [ [20, 90], [24, 82] ],
            [ [30, 90], [30, 82] ],
            [ [40, 90], [40, 82] ],
            [ [50, 90], [50, 82] ],
            [ [60, 90], [60, 82] ],
            [ [70, 90], [70, 82] ],
            [ [80, 90], [80, 82] ],
            [ [90, 78], [82, 78] ],
            [ [80, 66], [80, 74] ],
            [ [70, 66], [70, 74] ],
            [ [60, 66], [60, 74] ],
            [ [50, 66], [50, 74] ],
            [ [42, 66], [35, 79] ],
            [ [42, 66], [25, 69] ],
            [ [42, 62], [25, 59] ],
            [ [42, 62], [35, 49] ],
            [ [50, 54], [50, 62] ],
            [ [60, 54], [60, 62] ],
            [ [70, 54], [70, 62] ],
            [ [80, 54], [80, 62] ],
            [ [82, 54], [90, 54] ],
            [ [90, 47], [86, 47] ],
            [ [90, 36], [86, 36] ],
            [ [90, 25], [86, 25] ],
            [ [90, 14], [82, 18] ],
            [ [82, 10], [82, 18] ],
            [ [66, 18], [66, 26] ],
            [ [50, 10], [50, 18] ],
            [ [34, 18], [34, 26] ],
            [ [18, 10], [18, 18] ],
            [ [10, 14], [18, 18] ],
            [ [10, 20], [18, 20] ],

         ] )


    def find_ray_hit(self, start, ray_list):
        """ A function that uses the geometry package to find where the rays are
            comming into contact with the edges of the track
        """
        ## First we loop over the track segements to find if they are even in range of the car
        for track in [ self.inner_track, self.outer_track ]:
            for k in range(len(track)):
                l = (k+1) % (len(track))
                c = track[k]
                d = track[l]

                distance_to_car = Geometry.minimum_distance( c, d, start )
                if distance_to_car < self.max_ray_lenths:

                    for i in range(len(ray_list)):
                        intersect, loc = Geometry.find_intersection( start, ray_list[i], c, d )
                        if intersect:
                            ray_list[i] = loc

    def shoot_rays(self):
        """ A function that collects the distance to an object based
            for each forward ray
        """
        ## Next we need a list of the endpoints of each ray before they collide
        self.fwd_ray_end = np.array([ Geometry.rotate_2d_vec(  self.heading, a ) for a in self.fwd_ray_angles ])
        self.fwd_ray_end = self.max_ray_lenths * self.fwd_ray_end + self.car_front

        ## Now we find where the rays collide with the track
        self.find_ray_hit(self.car_front, self.fwd_ray_end)

        ## We also record the lengths of the list
        self.fwd_ray_lens = np.array([ np.linalg.norm(ray_end - self.car_front) for ray_end in self.fwd_ray_end ])

    def does_car_reach_new_gate(self):
        """ A function that checks if the car reached a NEW gate
            while updating the number of gates passed
        """

        c = self.reward_gates[self.n_gates_passed][0]
        d = self.reward_gates[self.n_gates_passed][1]

        ## The loop for car segments
        for i in range(4):
            j = (i+1) % 4
            a = self.car_v[i]
            b = self.car_v[j]
            test, _ = Geometry.find_intersection( a, b, c, d )
            if test:
                self.n_gates_passed += 1
                if self.n_gates_passed == len(self.reward_gates):
                    self.n_gates_passed = 0
                return True
        return False


    def does_car_touch_track(self):
        """ This function checks for line segment intersection, and it does so for
            every combination of car line and track line
        """

        ## First we loop over the track segements to find if they are even in range of the car
        for track in [ self.inner_track, self.outer_track ]:
            for k in range(len(track)):
                l = (k+1) % (len(track))
                c = track[k]
                d = track[l]
                distance_to_car = Geometry.minimum_distance( c, d, self.position )

                if distance_to_car < self.length*2:

                    ## The loop for car segments
                    for i in range(4):
                        j = (i+1) % 4
                        a = self.car_v[i]
                        b = self.car_v[j]
                        test, _ = Geometry.find_intersection( a, b, c, d )
                        if test:
                            return True

        return False


    def update_car_vectors(self):

        ## Getting various vectors which can point to different parts of the car
        self.side_heading = Geometry.rotate_2d_vec( self.heading, np.pi/2 )
        self.head_vec = self.length / 2 * self.heading
        self.side_vec = self.width  / 2 * self.side_heading

        self.car_front = self.position + self.head_vec
        self.car_back  = self.position - self.head_vec

        self.fwd_vel = np.dot(self.velocity, self.heading)
        self.sid_vel = np.dot(self.velocity, self.side_heading)

        ## The location of the car verticies (fl, fr, br, bl)
        self.car_v = np.array([
                     self.car_front + self.side_vec,
                     self.car_front - self.side_vec,
                     self.car_back  - self.side_vec,
                     self.car_back  + self.side_vec,
                     ])


    def get_traction( self ):
        speed = np.linalg.norm(self.velocity)
        if speed > self.slip_speed:
            return self.tracf
        else:
            return (self.tracf-self.tracs)/self.slip_speed * speed + self.tracs


    def decode_action(self, action):
        """ Decoding the action based on a scaler between 0 and 8
        """

        assert(action in range(12))
        if   action == 0:  return  1, 0,  1
        elif action == 1:  return  1, 0,  0
        elif action == 2:  return  1, 0, -1
        elif action == 3:  return  0, 0,  1
        elif action == 4:  return  0, 0,  0
        elif action == 5:  return  0, 0, -1
        elif action == 6:  return  1, 1,  1
        elif action == 7:  return  1, 1,  0
        elif action == 8:  return  1, 1, -1
        elif action == 9:  return  0, 1,  1
        elif action == 10: return  0, 1,  0
        elif action == 11: return  0, 1, -1


    def step(self, action):
        """ Moving ahead by one timestep using physics,
            then returning the new state
        """

        ## First we get the engine force and the turning angle from the user/network input
        self.fwd_state, self.brk_state, self.trn_state = self.decode_action(action)
        engine_mag = self.fwd_state * self.engine_max
        brake_mag  = self.brk_state * self.brake_max
        turn_angle = self.trn_state * self.turn_max

        ## Then we calculate the new positions of the wheels
        f_wheel  = self.position + self.wheel_base/2 * self.heading
        b_wheel  = self.position - self.wheel_base/2 * self.heading
        f_wheel += self.dt * Geometry.rotate_2d_vec( self.velocity, turn_angle )
        b_wheel += self.dt * self.velocity

        ## The new position of the car wheels determines the new heading
        self.heading = (f_wheel-b_wheel)/np.linalg.norm(f_wheel-b_wheel)

        ## The velocity rotated partly to this new heading with traction
        trac           = self.get_traction() / ( self.brake_mult if brake_mag > 0 else 1 )
        perf_vel       = self.heading * np.linalg.norm(self.velocity)
        self.velocity += ( perf_vel - self.velocity ) * trac

        ## Now we calculate the all forces acting on the car
        engine_force   =   engine_mag     * self.heading
        brake_force    = - brake_mag      * self.velocity
        friction_force = - self.fric_coef * self.velocity
        drag_force     = - self.drag_coef * self.velocity * np.linalg.norm(self.velocity)

        ## We then add up the forces to find the acceleration and the new velocity and position
        total_accel    = (engine_force + brake_force + friction_force + drag_force)/self.mass
        self.velocity += self.dt * total_accel
        self.position += self.dt * self.velocity
        self.time     += self.dt

        ## We also check if the car has come to a complete halt
        stalled = False
        if np.linalg.norm(self.velocity) < 5e-5:
            self.velocity = np.zeros(2)
            stalled = True

        ## We calculate the new position of the car and its vision rays
        self.update_car_vectors()
        self.shoot_rays()

        ## We then apply the collision mechanics to the car
        gate_hit  = self.does_car_reach_new_gate()
        track_hit = self.does_car_touch_track()

        ## We then calculate the reward of the new state
        reward = -0.01
        done = False
        if gate_hit:
            reward = 1.0
        if track_hit or stalled:
            reward = -1.0
            done = True

        ## If we reach the maximum timelimit then we end the episode with no reward
        if self.time >= self.time_limit:
            done = True

        info = 0
        return self.state, reward, done, info


    def render(self):
        """ Creating or updating the window to display the test
        """
        if self.viewer is None:
            self.viewer = CarGameWindow(self, 800, 800, "Car Test", resizable = False, vsync = False)
        self.viewer.render()



class CarGameWindow(pyglet.window.Window):
    """ The Window for visualising the driving model
    """
    def __init__(self, env, *args, **kwagrs):
        super().__init__(*args, **kwagrs)
        self.set_location( x=200, y=200 )

        self.keyboard = key.KeyStateHandler()
        self.env = env
        self.world_width = 200
        self.scale = self.width / self.world_width ## Converter from meters to pixels


    def on_close(self):
        quit()

    def draw_car(self):
        car_v_pixls = self.scale * self.env.car_v

        glColor4f(0.0, 0.0, 1.0, 1)
        glBegin(gl.GL_QUADS)
        for v in car_v_pixls:
            glVertex3f(*v, 0)
        glEnd()


    def draw_track(self, track):
        track_pixls = self.scale * track

        glLineWidth(3.0)
        glBegin(gl.GL_LINES)
        for i in range(len(track_pixls)):
            glColor4f(0,0,0,1)
            k = (i+1) % (len(track_pixls))
            glVertex2f( track_pixls[i][0], track_pixls[i][1] )
            glVertex2f( track_pixls[k][0], track_pixls[k][1] )
        glEnd()


    def draw_gates(self):
        gate_pxls = self.scale * self.env.reward_gates

        glColor4f(0.0, 1.0, 0.0, 1)
        glLineWidth(4.0)
        glBegin(gl.GL_LINES)
        for a,b in gate_pxls:
            glVertex2f( *a )
            glVertex2f( *b )
        if self.env.n_gates_passed>0:
            glColor4f(1.0, 0.0, 0.0, 1)
            glVertex2f( *gate_pxls[self.env.n_gates_passed-1][0] )
            glVertex2f( *gate_pxls[self.env.n_gates_passed-1][1] )
        glEnd()

    def draw_vision(self):

        y = self.scale*50
        x = self.scale*97
        fwd_ray_lens = 0.5*self.scale*self.env.fwd_ray_lens
        glColor4f(1.0, 0.0, 0.0, 1)
        glLineWidth(4.0)

        glBegin(gl.GL_LINES)
        for length in fwd_ray_lens:
            glVertex2f( x, y )
            glVertex2f( x, y+length )
            x = x - 4
        glEnd()

        fwd_vel = 3*self.scale*self.env.fwd_vel
        sid_vel = 3*self.scale*self.env.sid_vel
        y = self.scale*50
        x = self.scale*100
        glColor4f(1.0, 0.5, 0.5, 1)
        glLineWidth(4.0)
        glBegin(gl.GL_LINES)
        glVertex2f( x-2, y )
        glVertex2f( x-2, y+fwd_vel )
        glVertex2f( x+2, y )
        glVertex2f( x+2, y+sid_vel )
        glEnd()

    def draw_rays(self):
        car_front   = self.scale * self.env.car_front
        fwd_ray_end = self.scale * self.env.fwd_ray_end
        glColor4f(1.0, 0.0, 0.0, 1)
        glLineWidth(2.0)
        glBegin(gl.GL_LINES)
        for end in fwd_ray_end:
            glVertex2f( *car_front )
            glVertex2f( *end )
        glEnd()

    def draw_buttons(self):
        button_size = 50
        space = 10
        start_x = 500
        start_y = 300


        A_label = pyglet.text.Label('A',
                          font_size=36,
                          x=start_x+button_size/2, y=start_y+button_size/2,
                          anchor_x='center', anchor_y='center')
        start_x += button_size + space
        W_label = pyglet.text.Label('W',
                          font_size=36,
                          x=start_x+button_size/2, y=start_y+button_size/2,
                          anchor_x='center', anchor_y='center')
        start_x += button_size + space
        D_label = pyglet.text.Label('D',
                          font_size=36,
                          x=start_x+button_size/2, y=start_y+button_size/2,
                          anchor_x='center', anchor_y='center')
        start_x -= button_size + space
        start_y -= button_size + space
        B_label = pyglet.text.Label('B',
                          font_size=36,
                          x=start_x+button_size/2, y=start_y+button_size/2,
                          anchor_x='center', anchor_y='center')

        W_label.color = (0, 0, 255, 255) if self.env.fwd_state else (200, 200, 200, 255)
        A_label.color = (0, 0, 255, 255) if self.env.trn_state==+1 else (200, 200, 200, 255)
        D_label.color = (0, 0, 255, 255) if self.env.trn_state==-1 else (200, 200, 200, 255)
        B_label.color = (0, 0, 255, 255) if self.env.brk_state else (200, 200, 200, 255)

        A_label.draw()
        W_label.draw()
        D_label.draw()
        B_label.draw()

    def on_draw(self):
        """ Creating the new screen based on the new positions
        """
        glClearColor(1,1,1,1)
        self.clear()

        self.draw_track( self.env.outer_track )
        self.draw_track( self.env.inner_track )
        self.draw_car()
        self.draw_gates()
        self.draw_vision()
        self.draw_buttons()
        # self.draw_rays()


    def render(self):
        """ Update the ingame clock, gets the state from the simulation,
             and dispatches game events to render the car
        """
        pyglet.clock.tick()
        self.dispatch_events()
        self.dispatch_event('on_draw')
        self.flip()
