# Multi-View-MRI-Approach-for-Classification-of-MGMT-Methylation-in-Glioblastoma-Patients

# Introduction

This is an implementation of the model used for MGMT Methylation in Glioblastoma Patients classification status as described in our paper Multi View MRI Approach for Classification of MGMT Methylation in Glioblastoma Patients.In this work, we utilize Magnetic Resonance Imaging (MRI) scans along with deep learning models to detect MGMT methylation status. We present a novel approach that exploits the spatial dependencies between the the MRI views.

We provided training code , the Weights for trained models, as well as provided Comparison source code.



# Pipeline

The pre-processing pipeline consisting of three steps as shown in figure below:

- DICOM files were converted to NIFTI format for each modality
of each patient.
- Volumes were re-oriented to left-posterior-superior (LPS)
anatomical orientation. 
- Registered to SRI-24 human brain atlas is applied to provide a uniform anatomical coordinate system across all volumes.




# Architecture 
Below we show our approch that we used in the paper.
<div id="header" align="center">
  <img src="https://github.com/rawanalyahya/Multi-View-MRI-Approach-for-Classification-of-MGMT-Methylation-in-Glioblastoma-Patients/blob/main/figure/multi-view-model-v2.png" width="1000"/>
</div>

## Prerequisites:
- Python
- Torch
- Pydicom
- Pyradiomics 
