import numpy as np

''' Imitates a flock of birds behaviour'''


class Bird:
    def __init__(self, x, y, **kwargs):
        """
        Bird

        Parameters
        ----------
        x : x-coordinate of bird
        y : y-coordinate of bird

        **kwargs
        ----------
        vx : velocity_x
        vy : velocity_y
        radius : radius of visibility (default = 5)
        """
        self.id = None
        self.x = x
        self.y = y

        self.vx = kwargs.get('vx', None)
        self.vy = kwargs.get('vy', None)

        self.ax = kwargs.get('ax', None)
        self.ay = kwargs.get('ay', None)

        self.family = []
        # visibility radius of bird
        self.radius = 5  # units

        # Is the bird in sky?
        self.in_sky = False  # default = False

        if self.in_sky is True:
            self.avoid_nearby()
        # Let the bird look around the surroundings and

        self.avoid_nearby()  # returns a list of bird objects which are less than 5 units of distance

    def avoid_nearby(self):
        '''
        Identifies bird objects within a radius (self.radius)

        Note: No birds to be added while using this functions
        '''
        pass


class Rectangle:
    def __init__(self, x, y, w, h):
        """
        x = center of the Rectangle
        y = center of the Rectangle
        w = width of the rectangle
        h = height of the rectangle
        """
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.west_edge, self.east_edge = x - w / 2, x + w / 2
        self.north_edge, self.south_edge = y + h / 2, y - h / 2

    def contains(self, point):
        '''
        Checks if the point is in the boundary of the rectangle.


        '''
        return (point.x >= self.west_edge and point.x <= self.east_edge and
                point.y <= self.north_edge and point.y >= self.south_edge)


class Sky:
    """
    Creates an object where birds can exist.

    Parameters
    """

    def __init__(self, **kwargs):
        self.birds = []  # number of birds
        self.width = 100
        self.height = 100

        self.links = {}

    def create_grid(self, unit_length=1):
        """
        Creates Rectangle objects for each grid block.
        """
        blocks = []
        for i in np.arange(0, self.width + unit_length, unit_length):
            for j in np.arange(0, self.height + unit_length, unit_length):
                print(i, j)

    def insert(self, bird, **kwargs):
        """
        Adds a bird in the Sky.

        Parameters
        ----------
        bird : A Bird class object

        **kwargs
        ----------
        """
        if bird.id is None:
            # meaning, first bird
            bird.id = 1
        else:
            bird.id += 1

        # Let bird know he is in sky
        if bird.in_sky is False:
            bird.in_sky = True
            # Add bird to sky list
            self.birds.append(bird)
        else:
            print(f'Bird : {bird.id} already in Sky -_-')
            return

        # Update links:
        self.links[bird.id] = None


class TimeLoaf:
    def __init__(self, n_frames=30, **kwargs):
        """
        Creates a Time Loaf consisting of all time slices.

        Parameters
        ----------
        n_frames :
        """

        self.n_frames = n_frames

    def load(self, sky=None, n_frames=self.n_frames):
        pass

    def check_rules(self, ):
        """
        Returns the x1,y1, vx1,vy1, ax1,ay1 after checking rules
        """
        # Separation

        # Steer towards the average heading of the flock
        # Steer towards the average center of mass.
