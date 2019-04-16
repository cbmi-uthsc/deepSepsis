# This project can be divided into 3 Parts

## 1st Part  
 In this part we need to preprocess the data.  
 Acquire physiomarkers that are important for identifying the sepsis disease.  

 Since the data from CITI website can't be accessed due to unavailabitlity of login credential.  
 I can't find a good high frequency sepsis dataset online thus we need to skip this part for now.

 Lets assume we have the data set with all the physiological data as :
 [temperature, sao2, heartrate, respiration, cvp, etco2, systemicsystolic, systemicdiastolic, systemicmean, pasystolic, padiastolic, pamean,	st1, st2, st3, icp].  
 We will use various statistics method to get good physiomarkers for the dataset.  
 Lets assume these are the important physiomarkers for sepsis prediction [temperature, sao2, cvp, etco2, systemicystolic, systemicmean, st2,st2, icp].  

## 2nd Part

 In this part we will make different Temporal convolution neural network (t-CNN) architecture  with input data as physiomarkers that are acquired in phase 1.

 Training the architectures on train data.  
 Testing the architectures on test data.  
 Evaluating those architectures on the different metrics to identify the best performing architecture.  

## 3rd Part

 In this part we will Integrate the model to a python module.  
 Unit test the module created.
