import math
import time
import numpy as np

import torch
import torch.nn as nn

import pyglet
from pyglet.gl import *
from pyglet.window import key

class CartPoleEnv:
    """ The environment of the cart pole
    """
    def __init__(self):
        self.viewer = None

        ## The constants defining physics of the model
        self.dt        = 0.02
        self.gravity   = 9.8
        self.mass_cart = 1.0
        self.mass_pole = 0.1
        self.length    = 1.0
        self.force_mag = 10.0

        ## These variables which change also defining the state
        self.pos   = 0
        self.vel   = 0
        self.theta = 0
        self.omega = 0

        ## Derived quantaties
        self.total_mass          = self.mass_cart + self.mass_pole
        self.pole_halfmasslength = self.mass_pole * self.length / 2

        ## The failing conditions of the episode
        self.pos_fail   = 2.5
        self.theta_fail = 0.4
        self.time       = 0
        self.time_end   = 500

        ## The player action to make this interactive
        self.player_act = 0


    @property
    def state(self):
        return ( self.pos, self.vel, self.theta, self.omega )


    def reset(self, pos=0, vel=0, theta=0, omega=0):
        """ Resetting to the central vertical position
        """
        self.pos   = pos
        self.vel   = vel
        self.theta = theta
        self.omega = omega
        self.time  = 0


    def step(self, action):
        """ Moving ahead by one timestep using physics,
            then returning the new state
        """

        assert(action in [0,1])

        ## First we calculate the total force and its components
        force    = (1 if action == 1 else -1) * self.force_mag
        costheta = math.cos(self.theta)
        sintheta = math.sin(self.theta)

        temp = ( force + self.pole_halfmasslength * self.omega**2 * sintheta ) / self.total_mass

        ## Now we can calculate the acceneration of theta
        alpha = ( self.gravity * sintheta - costheta * temp ) / \
                ( self.length * ( 2.0 / 3.0 - self.mass_pole * costheta**2 / (2*self.total_mass) ) )

        ## And the acceleration of the cart
        accel = temp - self.pole_halfmasslength * alpha * costheta / self.total_mass

        ## Applying the update using the simplectic Euler method
        self.vel    += self.dt * accel
        self.omega  += self.dt * alpha
        self.pos    += self.dt * self.vel
        self.theta  += self.dt * self.omega

        ## We update the time
        self.time += 1

        ## We check if a terminal state has been reached
        failed = abs(self.pos) > self.pos_fail or abs(self.theta) > self.theta_fail or self.time > self.time_end
        failed = bool(failed)

        ## There is a reward for every timestep
        reward = 1

        return self.state, reward, failed

    def render(self):
        """ Creating or updating the window to display the test
        """
        if self.viewer is None:
            self.viewer = GameWindow(self, 600, 400, "Cart Pole", resizable = False, vsync = False)
        self.viewer.render(self.state)


class GameWindow(pyglet.window.Window):
    """ The Window for visualising the cartpole model
    """

    def __init__(self, parent, *args, **kwagrs):
        super().__init__(*args, **kwagrs)
        self.set_location( x=200, y=200 )

        self.parent = parent

    def on_close(self):
        quit()

    # def on_key_press(self, symbol, modifiers):
    #     if symbol == key.LEFT:
    #         self.parent.player_act = -1
    #     if symbol == key.RIGHT:
    #         self.parent.player_act = +1
    #
    # def on_key_release(self, symbol, modifiers):
    #     if symbol == key.LEFT or key.RIGHT:
    #         self.parent.player_act = 0

    def on_draw(self):
        """ Creating the new screen based on the new positions
        """
        glClearColor(1,1,1,1)
        self.clear()

        ## Calculating how to scale the system in the image
        world_width   = 5
        scale         = self.width/world_width

        ## The properties of the cart and pole in pixels
        pole_width  = 10
        pole_len    = self.parent.length * scale
        cart_width  = 120
        cart_height = 40
        cart_pos_x  = self.parent.pos * scale + self.width / 2
        cart_pos_y  = self.height / 3

        # Draw Track
        pyglet.gl.glColor4f( 0, 0, 0, 1)
        pyglet.gl.glLineWidth(2.0)
        pyglet.gl.glBegin(gl.GL_LINES)
        pyglet.gl.glVertex2f( 0,          cart_pos_y )
        pyglet.gl.glVertex2f( self.width, cart_pos_y )
        pyglet.gl.glEnd()

        ## Draw Cart
        l, r, t, b = (-cart_width/2, cart_width/2, cart_height/2, -cart_height/2)
        glColor4f(0.396, 0.263, 0.129, 1)
        glPushMatrix()
        glTranslatef(cart_pos_x, cart_pos_y, 0)
        glBegin(gl.GL_QUADS)
        glVertex3f(l, b, 0)
        glVertex3f(l, t, 0)
        glVertex3f(r, t, 0)
        glVertex3f(r, b, 0)
        glEnd()

        ## Draw Pole
        l, r, t, b = (-pole_width/2, pole_width/2, pole_len-pole_width/2, -pole_width/2)
        glColor4f(0, 0.5, 0.5, 1)
        glPushMatrix()
        glRotatef( -180/np.pi*self.parent.theta, 0, 0, 1.0)
        glBegin(gl.GL_QUADS)
        glVertex3f(l, b, 0)
        glVertex3f(l, t, 0)
        glVertex3f(r, t, 0)
        glVertex3f(r, b, 0)
        glEnd()

        glPopMatrix() ## The rotation matrix
        glPopMatrix() ## The translation matrix


    def render(self, state):
        """ Update the ingame clock, gets the state from the simulation,
             and dispatches game events to render the cart
        """
        pyglet.clock.tick()
        self.dispatch_events()
        self.dispatch_event('on_draw')
        self.flip()
