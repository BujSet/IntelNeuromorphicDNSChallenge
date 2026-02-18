import sofa, os
import scipy.io
import math
import numpy as np
import sys

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

        # Based on https://github.com/amini-allight/cipic-hrtf-database/blob/master/anthropometry/read_me.txt
        # lower indices are for left ear, upper are for right

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
        self._DL = D[0:8]
        self._DR = D[8:16]
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

    def chordDistBetweenIndices(self, srcIdx, destIdx):
        assert(0 <= srcIdx and srcIdx < 1250)
        assert(0 <= destIdx and destIdx < 1250)
        carts = self._sofa.Source.Position.get_values(system="cartesian")
        srcPos = carts[srcIdx]
        destPos = carts[destIdx]
        dist = np.linalg.norm(srcPos - destPos)
        return dist


    def getCartesianPositions(self):
        return  self._sofa.Source.Position.get_values(system="cartesian")

    def getSphericalPositionsFromIndex(self, index):
        sph_positions = self._sofa.Source.Position.get_values(system="spherical")

        # Only need azimuth and elevation
        sph_positions = np.delete(sph_positions, -1, 1)
        assert(0 <= index and index < 1250)
        pos = sph_positions[index]
        return  pos

    def getHRIRFromIndex(self, index, channel):
        vals = self._sofa.Data.IR.get_values()
        vals = vals[index]
        vals = vals[channel]
        return vals

    def collateAnthroData(self, ear="Right"):
        if ear =="Right":
            data = np.array(self._DR)
            data = np.append(data, self._rotationAngleR)
            data = np.append(data, self._flareAngleR)
        else:
            assert ear == "Left"
            data = np.array(self._DL)
            data = np.append(data, self._rotationAngleL)
            data = np.append(data, self._flareAngleL)
        return data

    def printAnthroData(self):
        string = "Subject " + str(self._id) + ":{\n"
        string += "\tAge:" + str(self._age) + "\n"
        if (abs(self._sex - 1.0) < 0.0001):
            string += "\tSex:M\n"
        elif(abs(self._sex - 2.0) < 0.0001):
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

    def __str__(self):
        string = "CipicSubject(sofaID=" + str(self._id)
        if (self._anthroDataIsComplete):
            string += ", Age=" + str(self._age)
            if (abs(self._sex - 1.0) < 0.0001):
                string += ", Sex=M"
            else:
                assert((abs(self._sex - 2.0) < 0.0001))
                string += ", Sex=F"
            string += ", Weight=" + str(self._weight)
            string += ")"
        return string

class CIPIC_DB():
    def __init__(self):
        self.cwd = os.path.join(os.getcwd(), "hrtfs")
        self.cwd = os.path.join(self.cwd, "cipic")
        self.subjects = dict()
        self.anthroData = scipy.io.loadmat('hrtfs/cipic/anthro.mat')

        subjectIDs = ["003","008","009","010","011",
                      "012","015","017","018","019",
                      "020","021","027","028","033",
                      "040","044","048","050","051",
                      "058","059","060","061","065",
                      "119","124","126","127","131",
                      "133","134","135","137","147",
                      "148","152","153","154","155",
                      "156","158","162","163","165"]
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

    def _getAnthroDataFromSubID(self, subID, field):
        # First we need to get the right Anthro ID, may be different from the 
        # SOFA ID.
        idx = 0
        for i in range(len(self.anthroData['id'])):
            sofaSub = int(self.anthroData['id'][i][0])
            if (sofaSub == subID):
                break
            idx += 1

        # Now we can read the important field
        fieldValue = self.anthroData[field][idx]
        if (len(fieldValue) == 0):
            return "NaN"
        elif (len(fieldValue) == 1):
            return fieldValue[0]
        else:
            return fieldValue

CipicDatabase = CIPIC_DB()
