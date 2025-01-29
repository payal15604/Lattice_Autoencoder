import prepare.generate_voxel_from_structure as gvfs
import prepare.lattice_autoencoder_old as la
import prepare.sites_autoencoder as sa
import prepare.sites_autoencoder_seperated as sas
import prepare.data_processing_for_GAN as dpfg
import prepare.train_constraint as tc
import gan.ccdcgan_git as gan
import prepare.data_transformation as dt
import os

##preprocess cif files  - doubt - don't know the fraction of atoms contriuting to each position whether it is 1 or less than one, would atoms not contrubuting fully be considered as 1 when calculating the stoichiometric sum?
#dt.process_cif_files2(text_directory='C:/Users/payal/OneDrive/Desktop/April Progress (CREEP)/cleaned_removed_data_6233.csv', cif_path='C:/Users/payal/OneDrive/Desktop/April Progress (CREEP)/CIF_FILES (8997)', data_path = 'C:/Users/payal/OneDrive/Desktop/April Progress (CREEP)/data', output_excel = './data.csv')
#training GAN with lr 0.0002 and iterations 1920
##### 1. generate voxels from crystal structures
print('generate voxels from crystal structures') #5881datapoints 8/10/24
#gvfs.voxel_generation(text_directory='C:/Users/payal/OneDrive/Desktop/April Progress (CREEP)/cleaned_removed_data_6233.csv',cif_path='C:/Users/payal/OneDrive/Desktop/April Progress (CREEP)/data',lattice_voxel_path='./calculation/payal_lattice_voxel_1010/',sites_voxel_path='./calculation/payal_sites_voxel_1010/',atomlisttype='all element',a_list=None, whether_lattice_voxel=True ,whether_sites_voxel=False)
#sa.find_best_autoencoder(sites_voxel_path='./calculation/payal_sites_voxel_testing/', model_folder_path='./calculation/autoencoder_model/sites/', encoded_sites_path='./calculation/payal_encoded_sites/', epochs=70, batch_size=512, train_ratio=0.8,validation_ratio=0.15,test_ratio=0.1,whether_training=False,whether_learning_rate_tune=False,learning_rate=0.01,whether_encoded_generate=False,whether_from_text=False,text_directory='C:/Users/payal/OneDrive/Desktop/April Progress (CREEP)/cleaned_removed_data_6233.csv',whether_voxel_generate=True, to_be_restored_encoded_sites_path='./calculation/payal_sites_voxel_testing_restored_og/', output_sites_voxel_path='./calculation/payal_restored_sites_voxel_generatemode_testing/', original_reconstruct = True)

##### 2. generate crystal images for crystal structures 
#### 2.1. train lattice autoencoder and generate encoded lattice 

#print('training lattice autoencoder')
#la.find_best_autoencoder(lattice_voxel_path='./calculation/payal_lattice_voxel_1010/', model_folder_path='./calculation/autoencoder_model/lattice_testing1311/', encoded_lattice_path='./calculation/payal_encoded_lattice_testing1311/', epochs=48, batch_size = 128, train_ratio=0.7,validation_ratio=0.15,test_ratio=0.15,whether_training=True,whether_learning_rate_tune=False,learning_rate=0.0003,whether_encoded_generate=True,whether_from_text=True,text_directory='C:/Users/payal/OneDrive/Desktop/April Progress (CREEP)/5881_data_csv.csv')dt.generated_lattice(generated_decoded_path='./calculation/payal_restored_lattice_voxel_testing1710_high/',generated_pre_path='./calculation/decoded_lattice_parameters/') #try for lr 0.01 85% accuracy batch size 
la.lattice_autoencoder(lattice_graph_path='./calculation/payal_lattice_voxel_1010/', encoded_graph_path='./calculation/payal_encoded_lattice_testing0801/', model_path='./calculation/autoencoder_model/lattice_testing0801/')
#la.restore_lattice_voxel_from_encoded_lattice(encoded_lattice_path = './calculation/payal_encoded_lattice_testing1311/', model_folder_path='./calculation/autoencoder_model/lattice_testing1311/', lattice_voxel_path = './calculation/payal_restored_lattice_voxel_testing1311/')
la.lattice_restorer(
    generated_2d_path='./calculation/payal_encoded_lattice_testing0801/',
    generated_decoded_path='./calculation/payal_restored_lattice_voxel_testing0801/',
    model_path='./calculation/autoencoder_model/lattice_testing0801/')
dt.generated_lattice(generated_decoded_path='./calculation/payal_restored_lattice_voxel_testing0801/',generated_pre_path='./calculation/decoded_lattice_parameters/')
#print('removing lattice voxel to release hard disk') 
#command='rm -r ./calculation/original_lattice_voxel/;'
#os.system(command)
print('Lattice Part Done')
#### 2.2. train general sites autoencoder and generate encoded sites
print('training sites autoencoder')
#sa.find_best_autoencoder(sites_voxel_path='./calculation/payal_sites_voxel/', model_folder_path='./calculation/autoencoder_model/sites/', encoded_sites_path='./calculation/payal_encoded_sites/', epochs=70, batch_size=512, train_ratio=0.8,validation_ratio=0.15,test_ratio=0.1,whether_training=False,whether_learning_rate_tune=False,learning_rate=0.01,whether_encoded_generate=False,whether_from_text=False,text_directory='C:/Users/payal/OneDrive/Desktop/April Progress (CREEP)/cleaned_data.csv',whether_voxel_generate=True, to_be_restored_encoded_sites_path='./calculation/payal_encoded_sites/', output_sites_voxel_path='./calculation/payal_output_sites_voxel/')
print('Sites Part done')
# 2.3. train elemental sites autoencoder and generate seperated encoded sites 
'''
train_list=[1, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 89, 90, 91, 92, 93, 94]
#train_list=[ 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 89, 90, 91, 92, 93, 94]
#train_list = [0]
#train_list = [41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 89, 90, 91, 92, 93, 94]
for element in train_list:
 print('element_number:',element)
 element_idx=element - 1
 sas.find_best_autoencoder(sites_voxel_path='./calculation/payal_sites_voxel/', model_folder_path='./calculation/autoencoder_model/sites_seperated/', encoded_sites_path='./calculation/payal_encoded_sites_seperated/', epochs=40, batch_size=512, train_ratio=0.8,validation_ratio=0.1,test_ratio=0.01,whether_training=False,whether_learning_rate_tune=False,learning_rate=0.01,whether_encoded_generate=False,whether_from_text=False,text_directory='C:/Users/payal/OneDrive/Desktop/April Progress (CREEP)/cleaned_data.csv',whether_voxel_generate=True,element_index=element_idx, output_sites_voxel_path = './calculation/payal_output_sites_voxel/', to_be_restored_encoded_sites_path = './calculation/payal_encoded_sites_seperated/' )
prnt('removing sites voxel to release hard disk')

#command='rm -r ./calculation/original_sites_voxel/;'
#os.system(command)                                                                             


#### 2.4. combine them as crystal images
print('HEADING : generating crystal images')

### 2.41 combining encoded lattice and encoded sites
print('combining encoded lattice and encoded sites')
dpfg.get_graph(text_directory = 'C:/Users/payal/OneDrive/Desktop/April Progress (CREEP)/cleaned_data.csv',encoded_lattice_path = './calculation/payal_encoded_lattice/',encoded_sites_path = './calculation/payal_encoded_sites/',graph_path = './calculation/payal_combined_graph/')
print('combined lattice and sites in a single .npy file')

### 2.42 rescale the images to 160 x 160 and create X .npy for training (redundant function)
dpfg.get_data_X(text_directory = 'C:/Users/payal/OneDrive/Desktop/April Progress (CREEP)/cleaned_data.csv',graph_path = './calculation/payal_combined_graph/' ,data_X_path = './calculation/data_to_train_GAN/',data_X_name = 'payal_combined_lattice_sites_X')


### 2.43 get large batch graph for X
dpfg.get_large_batch_graph(csv_path='C:/Users/payal/OneDrive/Desktop/April Progress (CREEP)/cleaned_data.csv',graph_path='./calculation/payal_combined_graph/',data_X_path='./calculation/data_to_train_GAN/',data_X_name='graphs_MP_20201009version_shuffled_other_primitive_cell_selected_cL10_aN20_no_inert.npy')

####command='rm -r ./calculation/original_encoded_sites/;'
##os.system(command)
#command='rm -r ./calculation/original_encoded_lattice/;'
#os.system(command)

##### 3. train constraint model
#### 3.1. get property as output
print('generating property vector')
dpfg.get_data_y(csv_directory='C:/Users/payal/OneDrive/Desktop/April Progress (CREEP)/filled_updated_data_cleaned.csv',property_name='bulk_modulus_hill',data_y_path='./calculation/data_to_train_constraint/',data_y_name='shuffled_bulk_modulus.npy')

dpfg.get_data_yx(csv_directory = 'C:/Users/payal/OneDrive/Desktop/April Progress (CREEP)/filled_updated_data_cleaned.csv', property_name = 'bulk_modulus_hill', data_y_path='./calculation/data_to_train_constraint/', data_y_name='shuffled_bulk_modulus_yx.npy', process_type='log10', graph_path='./calculation/payal_combined_graph/',data_X_path='./calculation/data_to_train_GAN/', data_X_name = 'graphs_MP_20201009version_shuffled_other_primitive_cell_selected_cL10_aN20_no_inert.npy')
#### 3.2. train constraint 

print('train constraint model')
tc.find_best_constrain_model(whether_from_trained_model=False, whether_training=True,whether_learning_rate_tune=False,learning_rate=0.0003,text_directory='C:/Users/payal/OneDrive/Desktop/April Progress (CREEP)/filled_updated_data_cleaned.csv', combined_graph_path='./calculation/data_to_train_GAN/', combined_graph_name='graphs_MP_20201009version_shuffled_other_primitive_cell_selected_cL10_aN20_no_inert.npy', property_directory='./calculation/data_to_train_constraint/shuffled_bulk_modulus_yx.npy',constrain_model_folder_path='./calculation/constrain_model/bulk_modulus/',constrain_model_name='bulk_modulus2')

##### 4. train CCDCGAN
print('train ccdcgan model')
ccdcgan = gan.CCDCGAN(whether_from_trained_model=False, whether_formation_energy_constrained=False,trained_formation_energy_directory='./calculation/constrain_model/bulk_modulus/best/bulk_modulus2.keras')
ccdcgan.train(text_directory='C:/Users/payal/OneDrive/Desktop/April Progress (CREEP)/filled_updated_data_cleaned.csv', combined_graph_path='./calculation/data_to_train_GAN/', combined_graph_name='graphs_MP_20201009version_shuffled_other_primitive_cell_selected_cL10_aN20_no_inert.npy', GAN_model_folder_path='./calculation/dcgan_model_git/', learning_process_text_name='DCGAN_learning_curve.txt')

for i in range(10):
 ccdcgan.predict(begin_epochs=5*i,epochs=5*i+5,generated_graph_folder_path='./dcgan_generated_graph/')
print('the training process has successfully finished')

#### 5. Restore encoded lattice and sites from generated new graphs:
dpfg.restore_encoded_lattice_sites_from_graph(graph_path = './dcgan_generated_graph/', whether_restore_encoded_lattice=True,encoded_lattice_path='./calculation/payal_restored_encoded_lattice/',whether_restore_encoded_sites=True,encoded_sites_path='./calculation/payal_restored_encoded_sites/') 
'''
#### 6. Restore lattice and sites voxels seperately from encoded data
## 6.1 Restore Lattice voxel from Lattice Encoded
#la.restore_lattice_voxel_from_encoded_lattice(encoded_lattice_path = './calculation/payal_encoded_lattice_largeratom/', model_folder_path='./calculation/autoencoder_model/lattice_largeratom/', lattice_voxel_path = './calculation/payal_restored_lattice_voxel_largeatom/')

## 6.2 Restore Sites voxel from Sites Encoded (Extract Information from encoded file)
#sa.find_best_autoencoder(sites_voxel_path='./calculation/payal_sites_voxel/', model_folder_path='./calculation/autoencoder_model/sites/', encoded_sites_path='./calculation/payal_encoded_sites/', epochs=70, batch_size=512, train_ratio=0.8,validation_ratio=0.15,test_ratio=0.1,whether_training=False,whether_learning_rate_tune=False,learning_rate=0.01,whether_encoded_generate=False,whether_from_text=False,text_directory='C:/Users/payal/OneDrive/Desktop/April Progress (CREEP)/cleaned_data.csv',whether_voxel_generate=True, to_be_restored_encoded_sites_path='./calculation/payal_restored_encoded_sites/', output_sites_voxel_path='./calculation/payal_restored_sites_voxel_generatemode/')
###

###6.3 restore lattice parameters from lattice voxel 
#dt.generated_lattice(generated_decoded_path='./calculation/payal_lattice_voxel_last_atom_49/',generated_pre_path='./calculation/decoded_lattice_parameters/')

