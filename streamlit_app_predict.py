import io
import streamlit as st
from roboflow import Roboflow
from pathlib import Path
import os
from PIL import Image
import cv2
import numpy as np
import base64





def load_image():
    opencv_image = None 
    path = None
    f = None
    uploaded_file = st.file_uploader(label='Pick an image to test')
    print(uploaded_file)
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        opencv_image = cv2.imdecode(file_bytes, 1)
        image_data = uploaded_file.getvalue() 
        #st.image(image_data)
        name = uploaded_file.name
        path = os.path.abspath(name)
        print("abs path")
        print(path)
	
        cv2.imwrite("main_image.jpg", opencv_image)
       
    return path, opencv_image
       

def loaddetDefectModel():
    print("...loading...Defect Check Model..")
    rf = Roboflow(api_key="kJNEcyxKtkAT9FmgW8x6")
    project = rf.workspace().project("brembo-bore-inspection-defect")
    model = project.version(1).model
    return model


# Take in base64 string and return PIL image
def stringToImage(base64_string):
    imgdata = base64.b64decode(base64_string)
    return Image.open(io.BytesIO(imgdata))

# convert PIL Image to an RGB image( technically a numpy array ) that's compatible with opencv
def toRGB(image):
    return np.array(image)
	

def detectDefect(cl, x, y, w, h, cnf, saved_image):
    print(".....inside detectDefect......")
    img = cv2.imread(saved_image)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    #print(img.shape)
    x = int(x)
    y = int(y)
    w = int(w)
    h = int(h)
    roi = img[y-h//2:y+h//2, x-w//2:x+w//2, :]
    st.image(roi, caption="ROI")
    cv2.imwrite("saved_ROI.jpg", roi)
    detDefect_model = loaddetDefectModel()
    defect_results = detDefect_model.predict("saved_ROI.jpg").json()
    print("defect detection results are....")
    print(defect_results)

    sd_image = roi.copy()
    for i in range(len(defect_results['predictions'])):
        x = defect_results['predictions'][i]['x']
        y = defect_results['predictions'][i]['y']
        w = defect_results['predictions'][i]['width']
        h = defect_results['predictions'][i]['height']
        cl = defect_results['predictions'][i]['class']
        cf = defect_results['predictions'][i]['confidence']

        sd_image = drawBoundingBoxDefect(roi,x,y,w,h,cl,cf)

    cv2.imwrite("final_res.jpg", sd_image)
    st.image(sd_image, caption="Defect Results")


def drawBoundingBoxDefect(saved_image ,x, y, w, h, cl, cf):
    #img = Image.open(saved_image)
    

    #img = cv2.imread(saved_image)
    img = cv2.cvtColor(saved_image,cv2.COLOR_BGR2RGB)
    x = int(x)
    y = int(y)
    w = int(w)
    h = int(h)
    start_pnt = (x-w//2,y-h//2)
    end_pnt = (x+w//2, y+h//2)
    txt_start_pnt = (x-w//2, y-h//2-15)
    
    
    img = cv2.rectangle(img, start_pnt, end_pnt, (255,0,0), 10)
    img = cv2.putText(img, cl, txt_start_pnt, cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 10, cv2.LINE_AA)		
    
    return img


def drawBoundingBox(saved_image ,x, y, w, h, cl, cf):
    #img = Image.open(saved_image)
    

    #img = cv2.imread(saved_image)
    img = cv2.cvtColor(saved_image,cv2.COLOR_BGR2RGB)
    x = int(x)
    y = int(y)
    w = int(w)
    h = int(h)
    start_pnt = (x-w//2,y-h//2)
    end_pnt = (x+w//2, y+h//2)
    txt_start_pnt = (x-w//2, y-h//2-15)
    
    
    img = cv2.rectangle(img, start_pnt, end_pnt, (0,255,0), 10)
    img = cv2.putText(img, cl, txt_start_pnt, cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 10, cv2.LINE_AA)	
    st.image(img, caption='Resulting Image')	
    


def predict(model, url):
    return model.predict(url, confidence=40, overlap=30).json()
    #return model.predict(url, hosted=True).json()
	
	
def main():
    st.title('BoreScope Inspection')
    image, svd_img = load_image()
    result = st.button('Predict')
    if result:
        st.write('Calculating results...')
        
	    #To Extract ROI
        rf = Roboflow(api_key="kJNEcyxKtkAT9FmgW8x6")
        project = rf.workspace().project("brembo-bore-inspection")
        model = project.version(1).model
	
        results = predict(model, svd_img)
    
        print("Prediction Results are...")	
        print(results)
        if len(results['predictions']) == 0:
            st.image(svd_img)
            st.write("No object is detected")
        else:
            new_img_pth = results['predictions'][0]['image_path']
            x = results['predictions'][0]['x']
            y = results['predictions'][0]['y']
            w = results['predictions'][0]['width']
            h = results['predictions'][0]['height']
            cl = results['predictions'][0]['class']
            cnf = results['predictions'][0]['confidence']
           
            print("printing saved image")
            #print(svd_img.name)
	
            #st.image(svd_img, "saved image")
            drawBoundingBox(svd_img,x, y, w, h, cl, cnf)
            #st.write(cl)
            #st.write(cnf)
            if(cl == "roi"):
                sem_seg_res = detectDefect(cl, x, y, w, h, cnf, "main_image.jpg")

   



    
    

if __name__ == '__main__':
    main()
