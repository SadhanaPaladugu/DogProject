import unittest
from app import *
class TestPredictImage(unittest.TestCase):
    APP_ROOT_FOLDER = os.path.dirname(os.path.abspath(__file__))
    def test_dog_detector(self):
        image1=os.path.join(APP_ROOT_FOLDER, '{}'.format("dogImages/train/021.Belgian_sheepdog/Belgian_sheepdog_01492.jpg"))
        image2=os.path.join(APP_ROOT_FOLDER, '{}'.format("dogImages/train/022.Belgian_tervuren/Belgian_tervuren_01563.jpg"))
        image3=os.path.join(APP_ROOT_FOLDER, '{}'.format("lfw/Ben_Howland/Ben_Howland_0001.jpg"))
        image4=os.path.join(APP_ROOT_FOLDER, '{}'.format("lfw/Ben_Howland"))
        image5=os.path.join(APP_ROOT_FOLDER, '{}'.format("test"))
      

        self.assertAlmostEqual(dog_detector(image1),True)
        self.assertAlmostEqual(dog_detector(image2),True)
        self.assertAlmostEqual(dog_detector(image3),False)
        self.assertAlmostEqual(dog_detector(image4),False)
        self.assertAlmostEqual(dog_detector(image5),False)
    def test_face_detector(self):
        image1=os.path.join(APP_ROOT_FOLDER, '{}'.format("dogImages/train/021.Belgian_sheepdog/Belgian_sheepdog_01492.jpg"))
        image2=os.path.join(APP_ROOT_FOLDER, '{}'.format("dogImages/train/022.Belgian_tervuren/Belgian_tervuren_01563.jpg"))
        image3=os.path.join(APP_ROOT_FOLDER, '{}'.format("lfw/Ben_Howland/Ben_Howland_0001.jpg"))
        image4=os.path.join(APP_ROOT_FOLDER, '{}'.format("lfw/Ben_Howland"))
        image5=os.path.join(APP_ROOT_FOLDER, '{}'.format("test"))
      

        self.assertAlmostEqual(face_detector(image1),False)
        self.assertAlmostEqual(face_detector(image2),False)
        self.assertAlmostEqual(face_detector(image3),True)
        self.assertAlmostEqual(face_detector(image4),False)
        self.assertAlmostEqual(face_detector(image5),False)
    def test_predictImage(self):
        image1=os.path.join(APP_ROOT_FOLDER, '{}'.format("dogImages/train/021.Belgian_sheepdog/Belgian_sheepdog_01492.jpg"))
        image2=os.path.join(APP_ROOT_FOLDER, '{}'.format("dogImages/train/022.Belgian_tervuren/Belgian_tervuren_01563.jpg"))
        image3=os.path.join(APP_ROOT_FOLDER, '{}'.format("lfw/Ben_Howland/Ben_Howland_0001.jpg"))
        
        model=load_ResNet50Model()
       

        _,result=predictImage(image1,model)
        self.assertAlmostEqual(result,"Belgian_sheepdog")

        _,result=predictImage(image2,model)
        self.assertAlmostEqual(result,"Belgian_tervuren")

        self.assertRaises(Exception, predictImage,image1,None)
        self.assertRaises(Exception, predictImage,image3,None)

       
        

        

