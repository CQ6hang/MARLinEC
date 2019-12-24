class User:
    def __init__(self, id, loc):
        self.user_id = id
        self.location = loc

    def print(self):
        print('user id:%s, user location:%s' % (self.user_id, self.location))
