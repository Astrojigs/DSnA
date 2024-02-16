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
        self.name = None
        self.x = x
        self.y = y

        # I AM SPEEEED
        self.vx = kwargs.get(vx)
        self.vy = kwargs.get(vy)

        # visibility radius of bird
        self.radius = 5 # units

        # Is the bird in sky?
        self.in_sky = False # default = False

        # Let the bird look around the surroundings and
        self.look_around() # returns a list of bird objects which are less than 5 units of distance


    def look_around(self):
        '''
        Looks around the circle with
        '''
        pass

class Sky:
    def __init__(self, **kwargs):
        self.birds = [] # number of birds
        self.width = 0
        self.height = 0

    def insert(self, bird, **kwargs):
        """
        Adds a bird in the Sky.

        Parameters
        ----------
        bird : A Bird class object

        **kwargs
        ----------
        """
        # change bird name to unique name
        if len(self.birds) != 0:
            if self.birds[-1].name is not None: # check if first bird name is not None.
                bird.name = self.birds[-1][:-1] + str(int(self.birds[-1][-1]) + 1) # change from 'sparrow_1' to 'sparrow_2'
            else: # set default name for first bird:
                self.birds[-1].name = 'sparrow_1'
        # Let bird know he is in sky
        if bird.in_sky is False:
            bird.in_sky = True
            # Add bird to sky list
            self.birds.append(bird)


        else:
            print(f'Bird : {bird.name} already in Sky -_-')

    def set_bounds(self, x, y):
        self.width = x
        self.height = y
