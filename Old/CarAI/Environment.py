import math
import time
import numpy as np

import Geometry
import torch
import torch.nn as nn

import pyglet
from pyglet.gl import *
from pyglet.window import key

class DrivingEnv:
    """ The environment of the car
    """
    def __init__(self):
        self.viewer = None

        ## The constants defining physics of the model
        self.time       = 0
        self.dt         = 0.2
        self.mass       = 1.0
        self.wheel_base = 3.0
        self.length     = 5.0
        self.width      = 3.0

        self.turn_max   = 0.3
        self.engine_max = 1.0
        self.brake_max  = 0.2

        self.fric_coef   = 0.01
        self.drag_coef   = 0.005

        self.slip_speed = 15
        self.tracs      = 0.3
        self.tracf      = 0.0001
        self.brake_mult = 10

        self.n_fwd_rays  = 17
        self.n_bck_rays  = 4

        self.fwd_state   = 0
        self.brk_state   = 0
        self.trn_state   = 0

        self.set_up_track()
        self.reset()

    @property
    def state(self):
        """ State returns a list of the various observables that will be used as inputs for the neural network
            These are the velocity projected along the heading axis
            The forward ray lengths
            The backward ray lengths
        """
        return np.concatenate( ([self.fwd_vel/10], [self.sid_vel/5], self.fwd_ray_lens/100-0.5, self.bck_ray_lens/100-0.5) )
        # return np.concatenate(  self.heading, , self.velocity/10 )

    def reset(self):
        """ Resetting to the central vertical position
        """
        self.position       = np.array([30.0,40.0])
        self.velocity       = np.zeros(2)
        self.heading        = Geometry.rotate_2d_vec( np.array([0,1]), np.random.uniform(-0.7, 0.7) )
        self.n_gates_passed = 0
        self.time = 0

        self.update_car_vectors()
        self.shoot_rays()

        return self.state

    def set_up_track(self):
        """Creating the track from a list of locations, the width of the playing world is 100
        """

        self.outer_track = 2*np.array( [ [ 10,  10 ],
                                         [ 10,  90 ],
                                         [ 90,  90 ],
                                         [ 90,  10 ] ] )


        self.inner_track = 2*np.array( [ [ 20,  20 ],
                                         [ 20,  80 ],
                                         [ 80,  80 ],
                                         [ 80,  20 ] ] )

        self.reward_gates = 2*np.array([ [ [10, 30], [20, 30] ],
                                         [ [10, 40], [20, 40] ],
                                         [ [10, 50], [20, 50] ],
                                         [ [10, 60], [20, 60] ],
                                         [ [10, 70], [20, 70] ],
                                         [ [10, 80], [20, 80] ],
                                         [ [20, 80], [20, 90] ],
                                         [ [30, 80], [30, 90] ],
                                         [ [40, 80], [40, 90] ],
                                         [ [50, 80], [50, 90] ],
                                         [ [60, 80], [60, 90] ],
                                         [ [70, 80], [70, 90] ],
                                         [ [80, 80], [80, 90] ],
                                         [ [80, 80], [90, 80] ],
                                         [ [80, 70], [90, 70] ],
                                         [ [80, 60], [90, 60] ],
                                         [ [80, 50], [90, 50] ],
                                         [ [80, 40], [90, 40] ],
                                         [ [80, 30], [90, 30] ],
                                         [ [80, 20], [90, 20] ],
                                         [ [80, 10], [80, 20] ],
                                         [ [70, 10], [70, 20] ],
                                         [ [60, 10], [60, 20] ],
                                         [ [50, 10], [50, 20] ],
                                         [ [40, 10], [40, 20] ],
                                         [ [30, 10], [30, 20] ],
                                         [ [20, 20], [20, 10] ], ] )


    def check_ray_hit(self, start, ray_list):
        for i in range(len(ray_list)):
            for track in [ self.inner_track, self.outer_track ]:
                for k in range(len(track)):
                    l = (k+1) % (len(track))
                    c = track[k]
                    d = track[l]
                    intersect, loc = Geometry.find_intersection( start, ray_list[i], c, d )
                    if intersect:
                        ray_list[i] = loc


    def shoot_rays(self):
        """ A function that collects the distance to an object based
            for each forward and backwards ray
        """

        ## Next we need a list of the endpoints of each ray
        self.fwd_ray_end  = np.array([ 100*Geometry.rotate_2d_vec(  self.heading, angle )+self.car_front for angle in np.pi/2*np.linspace(-0.9, 0.9, self.n_fwd_rays) ])
        self.bck_ray_end  = np.array([ 100*Geometry.rotate_2d_vec( -self.heading, angle )+self.car_back  for angle in np.pi/2*np.linspace(-0.9, 0.9, self.n_bck_rays) ])

        ## Now we cycle through the rays finding the line segments that are shortest
        self.check_ray_hit(self.car_front, self.fwd_ray_end)
        self.check_ray_hit(self.car_back,  self.bck_ray_end)

        ## We also record the lengths of the list
        self.fwd_ray_lens = np.array([ np.linalg.norm(ray_end - self.car_front) for ray_end in self.fwd_ray_end ])
        self.bck_ray_lens = np.array([ np.linalg.norm(ray_end - self.car_back)  for ray_end in self.bck_ray_end ])


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
            if Geometry.check_segment_intersect( a, b, c, d ):
                self.n_gates_passed += 1
                if self.n_gates_passed == len(self.reward_gates):
                    self.n_gates_passed = 0
                return True


    def does_car_touch_track(self):
        """ This function checks for line segment intersection, and it does so for
            every combination of car line and track line
        """

        ## The loop for car segments
        for i in range(4):
            j = (i+1) % 4
            a = self.car_v[i]
            b = self.car_v[j]

            for track in [ self.inner_track, self.outer_track ]:
                for k in range(len(track)):
                    l = (k+1) % (len(track))
                    c = track[k]
                    d = track[l]
                    if Geometry.check_segment_intersect( a, b, c, d ):
                        return True

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
        """ Decoding the action based on a scalaer between 0 and 8
        """

        assert(action in range(6))
        if   action == 0:  return  self.engine_max, 0,               self.turn_max
        elif action == 1:  return  self.engine_max, 0,               0
        elif action == 2:  return  self.engine_max, 0,              -self.turn_max
        elif action == 3:  return  0,               0,               self.turn_max
        elif action == 4:  return  0,               0,               0
        elif action == 5:  return  0,               0,              -self.turn_max
        # elif action == 6:  return  self.engine_max, self.brake_max,  self.turn_max
        # elif action == 7:  return  self.engine_max, self.brake_max,  0
        # elif action == 8:  return  self.engine_max, self.brake_max, -self.turn_max
        # elif action == 9:  return  0,               self.brake_max,  self.turn_max
        # elif action == 10: return  0,               self.brake_max,  0
        # elif action == 11: return  0,               self.brake_max, -self.turn_max


    def step(self, action):
        """ Moving ahead by one timestep using physics,
            then returning the new state
        """

        ## First we get the engine force and the turning angle from the user/network input
        engine_mag, brake_mag, turn_angle = self.decode_action(action)
        # engine_mag   = self.engine_max if self.fwd_state else 0
        # brake_mag    = self.brake_max  if self.brk_state else 0
        # turn_angle   = self.trn_state * self.turn_max

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
        reward = 0
        failed = False
        if gate_hit:
            reward = 1
        if track_hit or stalled:
            reward = -1
            failed = True

        return self.state, reward, failed

    def render(self):
        """ Creating or updating the window to display the test
        """
        if self.viewer is None:
            self.viewer = GameWindow(self, 800, 800, "Car Test", resizable = False, vsync = False)
        self.viewer.render(self.state)


class GameWindow(pyglet.window.Window):
    """ The Window for visualising the driving model
    """

    def __init__(self, parent, *args, **kwagrs):
        super().__init__(*args, **kwagrs)
        self.set_location( x=200, y=200 )

        self.keyboard = key.KeyStateHandler()
        self.parent   = parent

    def on_close(self):
        quit()

    def on_key_press(self, symbol, modifiers):
        if symbol == key.SPACE: self.parent.reset()
        if symbol == key.W: self.parent.fwd_state +=  1
        if symbol == key.B: self.parent.brk_state +=  1
        if symbol == key.A: self.parent.trn_state +=  1
        if symbol == key.D: self.parent.trn_state += -1

    def on_key_release(self, symbol, modifiers):
        if symbol == key.W: self.parent.fwd_state += -1
        if symbol == key.B: self.parent.brk_state += -1
        if symbol == key.A: self.parent.trn_state += -1
        if symbol == key.D: self.parent.trn_state +=  1


    def on_draw(self):
        """ Creating the new screen based on the new positions
        """

        glClearColor(1,1,1,1)
        self.clear()

        ## Calculating how to scale the system in the image
        world_width   = 200
        scale         = self.width/world_width ## Changes from meters to pixels

        ## Drawing the car based on its verticies
        car_v = scale*self.parent.car_v
        glColor4f(0.0, 0.0, 1.0, 1)
        glBegin(gl.GL_QUADS)
        for v in car_v:
            glVertex3f(*v, 0)
        glEnd()

        ## Drawing the track
        self.draw_track( scale*self.parent.outer_track )
        self.draw_track( scale*self.parent.inner_track )

        ## Drawing the reward_gates
        gate_lines = scale*self.parent.reward_gates
        glColor4f(0.0, 1.0, 0.0, 1)
        glLineWidth(4.0)
        glBegin(gl.GL_LINES)
        for a,b in gate_lines:
            glVertex2f( *a )
            glVertex2f( *b )
        if self.parent.n_gates_passed>0:
            glColor4f(0.0, 1.0, 1.0, 1)
            glVertex2f( *gate_lines[self.parent.n_gates_passed-1][0] )
            glVertex2f( *gate_lines[self.parent.n_gates_passed-1][1] )
        glEnd()

        ## Drawing the rays of the car
        # car_front   = scale*self.parent.car_front
        # car_back    = scale*self.parent.car_back
        # fwd_ray_end = scale*self.parent.fwd_ray_end
        # bck_ray_end = scale*self.parent.bck_ray_end
        # glColor4f(1.0, 0.0, 0.0, 1)
        # glLineWidth(2.0)
        # glBegin(gl.GL_LINES)
        # for end in fwd_ray_end:
            # glVertex2f( *car_front )
            # glVertex2f( *end )
        # for end in bck_ray_end:
            # glVertex2f( *car_back )
            # glVertex2f( *end )
        # glEnd()

        ## Drawing the ray mangitudes
        y = scale*45
        x = scale*90
        fwd_ray_lens = 0.4*scale*self.parent.fwd_ray_lens
        bck_ray_lens = 0.4*scale*self.parent.bck_ray_lens
        glColor4f(1.0, 0.0, 0.0, 1)
        glLineWidth(4.0)
        glBegin(gl.GL_LINES)
        for length in fwd_ray_lens:
            glVertex2f( x, y )
            glVertex2f( x, y+length )
            x = x - 4
        x = scale*110
        for length in bck_ray_lens:
            glVertex2f( x, y )
            glVertex2f( x, y+length )
            x = x + 4
        glEnd()

        ## Drawing the velocity magnidues mangitudes
        fwd_vel = 4*scale*self.parent.fwd_vel
        sid_vel = 4*scale*self.parent.sid_vel
        y = scale*60
        x = scale*100

        glColor4f(1.0, 0.5, 0.5, 1)
        glLineWidth(4.0)
        glBegin(gl.GL_LINES)
        glVertex2f( x, y )
        glVertex2f( x, y+fwd_vel )
        glVertex2f( x, y )
        glVertex2f( x-sid_vel, y )
        glEnd()




    def draw_track(self, track):
        glColor4f(0.0, 0.0, 0.0, 1)
        glLineWidth(3.0)
        glBegin(gl.GL_LINES)
        for i in range(len(track)):
            k = (i+1) % (len(track))
            glVertex2f( track[i][0], track[i][1] )
            glVertex2f( track[k][0], track[k][1] )
        glEnd()



    def render(self, state):
        """ Update the ingame clock, gets the state from the simulation,
             and dispatches game events to render the car
        """
        pyglet.clock.tick()
        self.dispatch_events()
        self.dispatch_event('on_draw')
        self.flip()






























