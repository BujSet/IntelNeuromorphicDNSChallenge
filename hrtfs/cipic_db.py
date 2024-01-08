import sofa, os

class CIPIC_Subject():
    def __init__(self, subID, filePath):
        print(filePath)
        self._id = subID
        self._sofa = sofa.Database.open(filePath)

    def __hash__(self):
        return hash(self._id)

    def getHRIRFromIndex(self, index, channel):
        vals = self._sofa.Data.IR.get_values()
        vals = vals[index]
        vals = vals[channel]
        return vals


class CIPIC_DB():
    def __init__(self):
        self.cwd = os.path.join(os.getcwd(), "hrtfs")
        self.cwd = os.path.join(self.cwd, "cipic")
        print(self.cwd)
        self.subjects = dict()
        subID = "012"
        filePath = os.path.join(self.cwd, "subject_" + subID + ".sofa")
        self.subjects[int(subID)] = CIPIC_Subject(int(subID), filePath)

CipicDatabase = CIPIC_DB();
print(CipicDatabase.subjects[12].getHRIRFromIndex(624, 0))
