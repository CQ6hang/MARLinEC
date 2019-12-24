import random


class BS:
    def __init__(self, id, loc, rds):
        self.site_id = id
        self.location = loc
        self.radius = rds

        self.type = random.randint(0, 2)

    def print(self):
        print('site id:%s, site location:%s, site signal radius:%d, site server:%d' % (
            self.site_id, self.location, self.radius, self.type))
