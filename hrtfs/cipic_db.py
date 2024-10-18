import sofa, os
import scipy.io
import math
import numpy as np

class CIPIC_Subject():
    def __init__(self, 
        subID, 
        sofaFilePath, 
        age,
        sex,
        weight,
        theta,
        X,
        D):
        self._id = subID
        self._sofa = sofa.Database.open(sofaFilePath)
        self._anthroDataIsComplete = True
        
        self._age = float(age)
        if math.isnan(self._age):
            self._anthroDataIsComplete = False

        self._sex = sex
        if (self._sex == "M"):
            self._sex = 1.0
        elif (self._sex == "F"):
            self._sex = 2.0
        else:
            self._sex = float('nan')
            self._anthroDataIsComplete = False

        self._weight = float(weight)
        if math.isnan(self._weight):
            self._anthroDataIsComplete = False

        self._rotationAngleL = float(theta[0])
        self._flareAngleL = float(theta[1])
        self._rotationAngleR = float(theta[2])
        self._flareAngleR = float(theta[3])
        for i in range(4):
            if (math.isnan(float(theta[i]))):
                self._anthroDataIsComplete = False
        self._X = X
        for i in range(17):
            if (math.isnan(float(X[i]))):
                self._anthroDataIsComplete = False
        self._DL = D[0:7]
        self._DR = D[8:15]
        for i in range(7):
            if (math.isnan(float(D[i]))):
                self._anthroDataIsComplete = False


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

    def collateAnthroData(self):
        data = np.array(self._X)
        data = np.append(data, self._age)
        data = np.append(data, [self._sex])
        data = np.append(data, [self._weight])
        data = np.append(data, self._DR)
        data = np.append(data, self._rotationAngleR)
        data = np.append(data, self._flareAngleR)

        sph_positions = self._sofa.Source.Position.get_values(system="spherical")
        sph_positions = np.delete(sph_positions, -1, 1)

        print(data)
        print(data.shape)
        print(sph_positions)
        print(sph_positions.shape)
        # self.printPositions()
        return data

    def printAnthroData(self):
        string = "Subject " + str(self._id) + ":{\n"
        string += "\tAge:" + str(self._age) + "\n"
        if (abs(self._sex) - 1.0 < 0.0001):
            string += "\tSex:M\n"
        elif(abs(self._sex) - 2.0 < 0.0001):
            string += "\tSex:F\n"
        else:
            string += "\tSex:-\n"
        string += "\tWeight:" + str(self._weight) + "\n"
        string += "\tRotation Angle {R, L}:" + str(self._rotationAngleR) + ", " + str(self._rotationAngleL) + "\n"
        string += "\tFlare Angle {R, L}:" + str(self._flareAngleR) + ", " + str(self._flareAngleL) + "\n"
        string += "\tX:" + str(self._X) + "\n"
        string += "\tD {R, L}:" + str(self._DR) + ", "+ str(self._DL) +"\n"
        string += "\tComplete:" + str(self._anthroDataIsComplete) + "\n"
        string += "}\n"
        print(string)


class CIPIC_DB():
    def __init__(self):
        self.cwd = os.path.join(os.getcwd(), "hrtfs")
        self.cwd = os.path.join(self.cwd, "cipic")
        self.subjects = dict()
        self.anthroData = scipy.io.loadmat('hrtfs/cipic/anthro.mat')
        # print(self.anthroData.keys())

        # TODO, really we should should just read the directory and see what
        # sofa files have been downloaded, but for now we'll just hardcode 
        # these
        subjectIDs = ["003", "008", "009", "010", "012", "017", "018", "021", "148", "165"]
        for subID in subjectIDs:
            filePath = os.path.join(self.cwd, "subject_" + subID + ".sofa")
            self.subjects[int(subID)] = CIPIC_Subject(
                int(subID), 
                filePath,
                self._getAnthroDataFromSubID(int(subID), 'age'), 
                self._getAnthroDataFromSubID(int(subID), 'sex'),
                self._getAnthroDataFromSubID(int(subID), 'WeightKilograms'),
                self._getAnthroDataFromSubID(int(subID), 'theta'),
                self._getAnthroDataFromSubID(int(subID), 'X'),
                self._getAnthroDataFromSubID(int(subID), 'D'))
            # self.subjects[int(subID)].printAnthroData()

    def _getAnthroDataFromSubID(self, subID, field):
        # First we need to get the right Anthro ID, may be different from the 
        # SOFA ID.
        idx = 0
        for i in range(len(self.anthroData['id'])):
            sofaSub = int(self.anthroData['id'][i][0])
            # print(sofaSub)
            if (sofaSub == subID):
                break
            idx += 1

        # print(self.anthroData["age"])
        # Now we can read the important field
        fieldValue = self.anthroData[field][idx]
        # print(fieldValue)
        if (len(fieldValue) == 0):
            return "NaN"
        elif (len(fieldValue) == 1):
            return fieldValue[0]
        else:
            return fieldValue



CipicDatabase = CIPIC_DB();
