import subprocess
import os
import zipfile

from flask import Flask, request, redirect, render_template, flash, send_file
from werkzeug.utils import secure_filename
from argparse import ArgumentParser
import pandas as pd
import torch
from torch.utils.data import DataLoader
import torch.multiprocessing
import pytorch_lightning as pl

from models.ROIPredictor_Multitask import RoiPredictor_Multitask
from datasets.ROIDataset import roi_dataset

torch.multiprocessing.set_sharing_strategy('file_system')
pl.seed_everything(42)
torch.backends.cudnn.determinstic = True
torch.backends.cudnn.benchmark = False

parser = ArgumentParser()

app = Flask(__name__)
app.secret_key = '123456789123456789'

trainer = pl.Trainer(logger=False)

model_path_amyloid = '/app/files/amyloid.ckpt'
model_path_gmvolume = '/app/files/gmvolume.ckpt'

model_amyloid = RoiPredictor_Multitask.load_from_checkpoint(model_path_amyloid)
model_gmvolume = RoiPredictor_Multitask.load_from_checkpoint(model_path_gmvolume)

dkt_labels = ["2","4","5","7","8","10","11","12","13","14","15","16","17","18","24","26","28","30","31","41","43","44","46","47","49","50","51","52","53","54","58","60","62","63","77","85","251","252","253","254","255","1000","1002","1003","1005","1006","1007","1008","1009","1010","1011","1012","1013","1014","1015","1016","1017","1018","1019","1020","1021","1022","1023","1024","1025","1026","1027","1028","1029","1030","1031","1034","1035","2000","2002","2003","2005","2006","2007","2008","2009","2010","2011","2012","2013","2014","2015","2016","2017","2018","2019","2020","2021","2022","2023","2024","2025","2026","2027","2028","2029","2030","2031","2034","2035"]
aal_labels = ["2001"," 2002"," 2101"," 2102"," 2111"," 2112"," 2201"," 2202"," 2211"," 2212"," 2301"," 2302"," 2311"," 2312"," 2321"," 2322"," 2331"," 2332"," 2401"," 2402"," 2501"," 2502"," 2601"," 2602"," 2611"," 2612"," 2701"," 2702"," 3001"," 3002"," 4001"," 4002"," 4011"," 4012"," 4021"," 4022"," 4101"," 4102"," 4111"," 4112"," 4201"," 4202"," 5001"," 5002"," 5011"," 5012"," 5021"," 5022"," 5101"," 5102"," 5201"," 5202"," 5301"," 5302"," 5401"," 5402"," 6001"," 6002"," 6101"," 6102"," 6201"," 6202"," 6211"," 6212"," 6221"," 6222"," 6301"," 6302"," 6401"," 6402"," 7001"," 7002"," 7011"," 7012"," 7021"," 7022"," 7101"," 7102"," 8101"," 8102"," 8111"," 8112"," 8121"," 8122"," 8201"," 8202"," 8211"," 8212"," 8301"," 8302"," 9001"," 9002"," 9022"," 9031"," 9032"," 9041"," 9042"," 9120"]
labels_mean_2 = ["3000","3001","3002","3003","3004","3005","3006","3007","3008","3009","3010","3011","3012","3013","3014","3015","3016","3017","3018","3019","3020","3021","3022","3023","3024","3025","3026","3027","3028","3029","3030","3031","3032","3033","3034","3035"]
labels_mean_41 = ["4000","4001","4002","4003","4004","4005","4006","4007","4008","4009","4010","4011","4012","4013","4014","4015","4016","4017","4018","4019","4020","4021","4022","4023","4024","4025","4026","4027","4028","4029","4030","4031","4032","4033","4034","4035"]

def encode_apoe_one_hot(apoe):
  if apoe == '33':
    return '1,0,0,0,0,0'
  if apoe == '34':
    return '0,1,0,0,0,0'
  if apoe == '44':
    return '0,0,1,0,0,0'
  if apoe == '24':
    return '0,0,0,1,0,0'
  if apoe == '23':
    return '0,0,0,0,1,0'
  if apoe == '22':
    return '0,0,0,0,0,1'
  raise ValueError(f'apoe value {apoe} not recognized')

def encode_m_f(sex):
  if sex == 'male':
    return '0,1'
  else:
    return '1,0'

def prepend_tabular(csv_path, sex, age, apoe):
    with open(csv_path, 'r') as f:
      _ = f.readline() # Throw away header specifying atlas regions
      csv = f.readline()
    out = f'{sex},{age},{apoe},{csv}'
    with open(csv_path, 'w') as f:
      f.write(out)

def fix_dkt(csv_path, file_base):
  all_df = pd.read_csv(csv_path)
  labels_mean_2_intersection = all_df.columns.intersection(labels_mean_2)
  labels_mean_41_intersection = all_df.columns.intersection(labels_mean_41)
  val_mean_2 = all_df[labels_mean_2_intersection].mean(axis=1)
  val_mean_41 = all_df[labels_mean_41_intersection].mean(axis=1)
  all_df['2'] = val_mean_2
  all_df['41'] = val_mean_41
  all_df['4'] = 1
  all_df['43'] = 1
  all_df['44'] = 1
  filter_df = all_df[dkt_labels]
  filter_df.to_csv(csv_path, index=False)
  filter_df['T+'] = int(filter_df.mean(axis=1)>1.3)
  filter_df = filter_df[ ['T+'] + [ col for col in filter_df.columns if col != 'T+' ] ]
  filter_df.to_csv(os.path.join('/app/files',file_base+'_tau_dkt.csv'), index=False)

@app.route('/')
def home():
  return render_template('web.html')

@app.route('/predict', methods=['POST'])
def predict():
  sex = encode_m_f(request.form["sex"])
  try:
    float(request.form['age'])
  except ValueError:
    flash('Age must be a number', 'error')
    return redirect('/')
  age = float(request.form["age"])-75
  apoe = encode_apoe_one_hot(request.form["apoe"])
  if 'tau_pet' not in request.files:
    print('No tau_pet file')
    return redirect('/')
  tau_pet_file = request.files['tau_pet']
  try:
    assert tau_pet_file.filename.endswith('nii.gz') or tau_pet_file.filename.endswith('nii')
  except AssertionError:
    flash("Please upload a .nii or .nii.gz file", 'error')
    return redirect('/')
  tau_file_path = os.path.join('/app/files',secure_filename(tau_pet_file.filename))
  tau_pet_file.save(tau_file_path)
  file_base = tau_pet_file.filename.split('.')[0]

  rc = subprocess.call(f"Rscript /app/files/Extract_PET_ROIs_from_atlas.R -m /app/files/GM_mask.nii.gz -a /app/files/Schaefer2018_200Parcels_7Networks_order_FSLMNI152_2mm.nii.gz -p {tau_file_path} -o /app/files/pet_schaefer.csv", shell=True)
  try:
    assert rc==0, "Failed to extract schaefer regions from PET. Check input arguments"
  except AssertionError as e:
    flash(e, 'error')
    return redirect('/')
  
  rc = subprocess.call(f"Rscript /app/files/Extract_PET_ROIs_from_atlas.R -m /app/files/GM_mask.nii.gz -a /app/files/rDesikan-Killiany_MNI_SPM12.nii.gz -p {tau_file_path} -o /app/files/pet_dkt.csv", shell=True)
  try:
    assert rc==0, "Failed to extract dkt regions from PET. Check input arguments"
  except AssertionError as e:
    flash(e, 'error')
    return redirect('/')

  fix_dkt("/app/files/pet_dkt.csv", file_base)
  prepend_tabular("/app/files/pet_dkt.csv",sex,age,apoe)
  prepend_tabular("/app/files/pet_schaefer.csv",sex,age,apoe)

  # Amyloid prediction 
  data_loader_amyloid = DataLoader(roi_dataset('/app/files/pet_dkt.csv'), batch_size=1, shuffle=False, num_workers=1)
  preds_amyloid = trainer.predict(model_amyloid, data_loader_amyloid)[0]
  regional_amyloid = preds_amyloid[0]
  amyloid_positivity_pred = int((torch.softmax(preds_amyloid[1], dim=1)>0.2663).flatten()[1].numpy())
  regional_amyloid = regional_amyloid.flatten().numpy()
  regional_amyloid_outfile_path = os.path.join('/app/files',file_base+'_amyloid_dkt.csv')
  with open(regional_amyloid_outfile_path, 'w') as f:
    f.write('A+,')
    f.write(','.join([x for x in dkt_labels]))
    f.write('\n')
    f.write(str(amyloid_positivity_pred)+',')
    f.write(','.join([str(x) for x in regional_amyloid]))

  # Nuerodegeneration prediction
  data_loader_gmvolume = DataLoader(roi_dataset('/app/files/pet_schaefer.csv'), batch_size=1, shuffle=False, num_workers=1)
  preds_gmvolume = trainer.predict(model_gmvolume, data_loader_gmvolume)[0]
  regional_gmvolume = preds_gmvolume[0]
  neurodegen_pred = int((torch.softmax(preds_gmvolume[1], dim=1)>0.2486).flatten()[1].numpy())
  regional_gmvolume = regional_gmvolume.flatten().numpy()
  regional_gmvolume_outfile_path = os.path.join('/app/files',file_base+'_neurodegen_gmvolume_AAL.csv')
  with open(regional_gmvolume_outfile_path, 'w') as f:
    f.write('N+,')
    f.write(','.join([x for x in aal_labels]))
    f.write('\n')
    f.write(str(neurodegen_pred)+',')
    f.write(','.join([str(x) for x in regional_gmvolume]))

  # Tau
  regional_tau_outfile_path = os.path.join('/app/files',file_base+'_tau_dkt.csv')

  # ZIP all results
  zip_path = os.path.join('/app/files',file_base+'_results_amyloid_tau_neurodegen.zip')
  with zipfile.ZipFile(zip_path, 'w') as zipObj:
    zipObj.write(regional_amyloid_outfile_path, arcname=os.path.basename(regional_amyloid_outfile_path))
    zipObj.write(regional_gmvolume_outfile_path, arcname=os.path.basename(regional_gmvolume_outfile_path))
    zipObj.write(regional_tau_outfile_path, arcname=os.path.basename(regional_tau_outfile_path))

  return send_file(zip_path, as_attachment=True)

if __name__ == '__main__':
  app.run(host='0.0.0.0', port=5000)