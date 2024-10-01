import Animal

class Troop(object):

    def __init__(self, dsinfo):

        super().__init__()

        # Bind critical identifying information
        self.investigators = dsinfo['investigators']
        self.names         = dsinfo['names']
        self.areas         = dsinfo['areas']
        self.cat_labels    = dsinfo['cat_labels']
        self.dir_labels    = dsinfo['dir_labels']
        self.dbounds       = dsinfo['dbounds']
        self.dicd_splits   = dsinfo['dicd_splits']

        # Form array of Animal() objects, one per specified animal
        self.animals = []
        for i,k in enumerate(self.names):
            self.animals.append(
                Animal.Animal(k, 
                              self.investigators[i],
                              self.areas, 
                              self.cat_labels[k], 
                              self.dir_labels[k], 
                              self.dbounds[k], 
                              self.dicd_splits[k])
            )

        return
