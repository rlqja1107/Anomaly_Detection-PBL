# Anomaly Detection  

## 설명  
MNIST Data에서 숫자 0인 이미지를 Noraml로 보고 숫자 1~9인 이미지를 Anomaly라고 생각하여 Test시에 숫자 0인 것과 0이 아닌 것을 Detect하는지 확인하고자 한다. 아래의 모델를 이용하여 서로 비교하고자 한다.

**Anogan** 
**OneClassSVM**  
**DeepSVDD**  
**Autoencoder**  

### 추가 실험    

Data Augmentation을 했을 때 모델에서 Normal Data의 Feature를 더 잘 Capture하고 성능에서의 차이점이 존재하는지 확인하고자 한다.  

다양한 실험 셋

### 실험 셋팅  

|실험 번호|비율|Train Normal|Train Abnormal|Test Normal|Test Abnormal|  
|:---:|:---:|:---:|:---:|:---:|:---:|  
|1|10:1|5923|593|980|99|   
|1|100:1|5923|60|980|10|  
|1|10:1 (x10)|59230|5924|980|99|  
|1|10:1 (x10)|59230|593|980|10|  

### 실험 결과  
* 실험 1  

|Model|Precision|Recall|F1 Score|AUC|   
|:---:|:---:|:---:|:---:|:---:|    
|Autoencoder|0.99|0.89|0.94|0.944|   
|OneSVM|||||    
|DeepSVD|1|1|1|1|  
|AnoGAN|0.996|0.913265|0.952128|0.960555|  

* 실험 3  

|Model|Precision|Recall|F1 Score|AUC|   
|:---:|:---:|:---:|:---:|:---:|    
|Autoencoder|0.98|0.83|0.9|0.901|   
|OneSVM|||||    
|DeepSVD|1|1|1|0.997|  
|AnoGAN|0.989|0.882|0.933|0.93|  


