import sofa, os

class CIPIC_Subject():
    def __init__(self, subID, filePath):
        self._id = subID
        self._sofa = sofa.Database.open(filePath)

    def __hash__(self):
        return hash(self._id)

    def printPositions(self):
        cart_positions = self._sofa.Source.Position.get_values(system="cartesian")
        sph_positions = self._sofa.Source.Position.get_values(system="spherical")
        for i in range(len(cart_positions)):
            pos = cart_positions[i]
            print(str(i) + str(pos) + str(sph_positions[i]))

    def getCartesianPositions(self):
        return  self._sofa.Source.Position.get_values(system="cartesian")

    def getHRIRFromIndex(self, index, channel):
        vals = self._sofa.Data.IR.get_values()
        vals = vals[index]
        vals = vals[channel]
        return vals


class CIPIC_DB():
    def __init__(self):
        self.cwd = os.path.join(os.getcwd(), "hrtfs")
        self.cwd = os.path.join(self.cwd, "cipic")
        self.subjects = dict()
        # TODO, really we should should just read the directory and see what
        # sofa files have been downloaded, but for now we'll just hardcode 
        # these
        subjectIDs = ["012", "021", "165"]
        for subID in subjectIDs:
            filePath = os.path.join(self.cwd, "subject_" + subID + ".sofa")
            self.subjects[int(subID)] = CIPIC_Subject(int(subID), filePath)

CipicDatabase = CIPIC_DB();
