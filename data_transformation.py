import os

from ase.io import read, write
from ase import Atom, Atoms

import numpy as np
from numpy.linalg import norm
import random
import pandas as pd
from joblib import Parallel, delayed

from scipy.ndimage import maximum_filter, gaussian_filter, minimum_filter
from scipy.ndimage.morphology import generate_binary_structure, binary_erosion

import math

elements_with_index = [
    (1, 'H'), (2, 'He'), (3, 'Li'), (4, 'Be'), (5, 'B'), (6, 'C'), (7, 'N'), (8, 'O'), (9, 'F'), (10, 'Ne'),
    (11, 'Na'), (12, 'Mg'), (13, 'Al'), (14, 'Si'), (15, 'P'), (16, 'S'), (17, 'Cl'), (18, 'Ar'), (19, 'K'), (20, 'Ca'),
    (21, 'Sc'), (22, 'Ti'), (23, 'V'), (24, 'Cr'), (25, 'Mn'), (26, 'Fe'), (27, 'Co'), (28, 'Ni'), (29, 'Cu'), (30, 'Zn'),
    (31, 'Ga'), (32, 'Ge'), (33, 'As'), (34, 'Se'), (35, 'Br'), (36, 'Kr'), (37, 'Rb'), (38, 'Sr'), (39, 'Y'), (40, 'Zr'),
    (41, 'Nb'), (42, 'Mo'), (43, 'Tc'), (44, 'Ru'), (45, 'Rh'), (46, 'Pd'), (47, 'Ag'), (48, 'Cd'), (49, 'In'), (50, 'Sn'),
    (51, 'Sb'), (52, 'Te'), (53, 'I'), (54, 'Xe'), (55, 'Cs'), (56, 'Ba'), (57, 'La'), (58, 'Ce'), (59, 'Pr'), (60, 'Nd'),
    (61, 'Pm'), (62, 'Sm'), (63, 'Eu'), (64, 'Gd'), (65, 'Tb'), (66, 'Dy'), (67, 'Ho'), (68, 'Er'), (69, 'Tm'), (70, 'Yb'),
    (71, 'Lu'), (72, 'Hf'), (73, 'Ta'), (74, 'W'), (75, 'Re'), (76, 'Os'), (77, 'Ir'), (78, 'Pt'), (79, 'Au'), (80, 'Hg'),
    (81, 'Tl'), (82, 'Pb'), (83, 'Bi'), (84, 'Po'), (85, 'At'), (86, 'Rn'), (87, 'Fr'), (88, 'Ra'), (89, 'Ac'), (90, 'Th'),
    (91, 'Pa'), (92, 'U'), (93, 'Np'), (94, 'Pu'), (95, 'Am'), (96, 'Cm'), (97, 'Bk'), (98, 'Cf'), (99, 'Es'), (100, 'Fm'),
    (101, 'Md'), (102, 'No'), (103, 'Lr'), (104, 'Rf'), (105, 'Db'), (106, 'Sg'), (107, 'Bh'), (108, 'Hs'), (109, 'Cn'), (110, 'Nh'),
    (111, 'Fl'), (112, 'Mc'), (113, 'Lv'), (114, 'Ts'), (115, 'Og'), (116, 'Lv'), (117, 'Ts'), (118, 'Og')
]




#####define the used atom list
def get_atomlist_atomindex(atomlisttype='all element',a_list=None):#atomlisttype indicate which kind of list to be used; 'all element' is to use the whole peroidic table, 'specified' is to give a list on our own
	if atomlisttype=='all element':
		all_atomlist = ['H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar',
		          'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 
		          'Kr', 'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Te', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 
		          'I', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 
		          'Yb', 'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', 
		          'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm','Md', 'No', 'Lr',
		          'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg', 'Cn', 'Nh', 'Fl', 'Mc', 'Lv', 'Ts', 'Og', 'Uue']
	else:
		if atomlisttype=='specified':
			all_atomlist=a_list
		else:
			print('atom list type is not acceptable')
			return
	cod_atomlist=all_atomlist
	cod_atomindex = {}
	for i,symbol in enumerate(all_atomlist):
		cod_atomindex[symbol] = i

	return cod_atomlist,cod_atomindex

def get_scale(sigma):
	scale = 1.0/(2*sigma**2)
	return scale

def get_atoms(inputfile,filetype):
	atoms = read(inputfile,format = filetype)
	return atoms

def extract_cell(atoms):
	cell = atoms.cell
	atoms_ = Atoms('Bi')
	atoms_.cell = cell
	atoms_.set_scaled_positions([0.5,0.5,0.5])
	return atoms_

def get_fakeatoms_grid(atoms,nbins):
	atomss = []
	scaled_positions = []
	ijks = []
	grid = np.array([float(i)/float(nbins) for i in range(nbins)])
	yv,xv,zv = np.meshgrid(grid,grid,grid)
	pos = np.zeros((nbins**3,3))
	pos[:,0] = xv.flatten()
	pos[:,1] = yv.flatten()
	pos[:,2] = zv.flatten()
	atomss = Atoms('H'+str(nbins**3))
	atomss.set_cell(atoms.get_cell())#making pseudo-crystal containing H positioned at pre-defined fractional coordinate
	atomss.set_pbc(True)
	atomss.set_scaled_positions(pos)
	fakeatoms_grid = atomss
	return fakeatoms_grid

def get_image_one_atom(atom,fakeatoms_grid,nbins,scale):
	grid_copy = fakeatoms_grid.copy()
	ngrid = len(grid_copy)
	image = np.zeros((1,nbins**3))
	grid_copy.append(atom)
	drijk = grid_copy.get_distances(-1,range(0,nbins**3),mic=True)
	pijk = np.exp(-scale*drijk**2)
	image[:,:] = pijk.flatten()
	return image.reshape(nbins,nbins,nbins)

def get_image_all_atoms(atoms,nbins,scale,norm,num_cores,atomlisttype,a_list):
	fakeatoms_grid = get_fakeatoms_grid(atoms,nbins)
	cell = atoms.get_cell()
	imageall_gen = Parallel(n_jobs=num_cores)(delayed(get_image_one_atom)(atom,fakeatoms_grid,nbins,scale) for atom in atoms)
	imageall_list = list(imageall_gen)
	cod_atomlist,cod_atomindex = get_atomlist_atomindex(atomlisttype,a_list)
	nchannel = len(cod_atomlist)
	channellist = []
	for i,atom in enumerate(atoms):
		channel = cod_atomindex[atom.symbol]
		channellist.append(channel)
	channellist = list(set(channellist))
	nc = len(channellist)
	shape = (nbins,nbins,nbins,nc)
	image = np.zeros(shape,dtype=np.float32)
	for i,atom in enumerate(atoms):
		nnc = channellist.index(cod_atomindex[atom.symbol])
		img_i = imageall_list[i]
		image[:,:,:,nnc] += img_i * (img_i>=0.02)
		 
	return image,channellist


def basis_translate(atoms):
	N = len(atoms)
	pos = atoms.positions
	cg = np.mean(pos,0)
	dr = 7.5 - cg #move to center of 15A-cubic box
	dpos = np.repeat(dr.reshape(1,3),N,0)
	new_pos = dpos + pos
	atoms_ = atoms.copy()
	atoms_.cell = 15.0*np.identity(3)
	atoms_.positions = new_pos
	return atoms_

def generate_sites_graph(sites_graph_path,atomlisttype,a_list,data_path,data_type):#e.g.: dt.generate_sites_graph(sites_graph_path='./original_lattice_graph/',atomlisttype='specified',a_list=['V','O'],data_path='/home/teng/tensorflow2.0_example/imatgen-master/iMatGen-VO_dataset_generated_strctures/VO_dataset/geometries/',data_type='vasp')
	if not os.path.exists(sites_graph_path):
		os.makedirs(sites_graph_path)

	scale=get_scale(0.26)
	filename=os.listdir(data_path)#'./TC/chem_info/')

	for eachfile in filename:
		if eachfile.endswith(data_type):
			filename=data_path+eachfile
			atoms=get_atoms(filename,data_type)
			atoms_=basis_translate(atoms)
			image,channellist=get_image_all_atoms(atoms_,64,scale,norm,8,atomlisttype,a_list)

			savefilename=sites_graph_path+eachfile[:-len(data_type)-1]+'.npy'
			np.save(savefilename,image)

def generate_combined_sites_graph(sites_graph_path,atomlisttype,a_list,data_path,data_type):
	if not os.path.exists(sites_graph_path):
		os.makedirs(sites_graph_path)

	scale=get_scale(0.26)
	filename=os.listdir(data_path)

	for eachfile in filename:
		if eachfile.endswith(data_type):
			filename=data_path+eachfile
			print(filename)
			atoms=get_atoms(filename,data_type)
			atoms_=basis_translate(atoms)
			image,channellist=get_image_all_atoms(atoms_,64,scale,norm,8,atomlisttype,a_list)

			_,_,_,nc=image.shape
			combined_image=np.zeros([64,64,64])
			for i in range(nc):
				combined_image=combined_image+image[:,:,:,i]

			savefilename=sites_graph_path+eachfile[:-len(data_type)-1]+'.npy'
			np.save(savefilename,combined_image)

def generate_lattice_graph(lattice_graph_path,atomlisttype,a_list,data_path,data_type):#e.g.: dt.generate_lattice_graph(lattice_graph_path='./original_lattice_graph/',atomlisttype='specified',a_list=['V'],data_path='/home/teng/tensorflow2.0_example/imatgen-master/iMatGen-VO_dataset_generated_strctures/VO_dataset/geometries/',data_type='vasp')
	if not os.path.exists(lattice_graph_path):
		os.makedirs(lattice_graph_path)

	scale=get_scale(0.26)
	filename=os.listdir(data_path)
    
	for eachfile in filename:
		if eachfile.endswith(data_type):

			filename=data_path+eachfile
			print(filename)
			atoms=get_atoms(filename,data_type)
			atoms_=extract_cell(atoms)
			image,channellist=get_image_all_atoms(atoms_,32,scale,norm,8,atomlisttype,a_list)
			image=image.reshape(32,32,32)

			savefilename=lattice_graph_path+eachfile[:-len(data_type)-1]+'.npy'
			np.save(savefilename,image)

def generate_crystal_2d_graph(encodedgraphsavepath='./encoded_sites/',encodedlatticesavepath='./encoded_lattice/',crystal_2d_graph_path='./crystal_2d_graphs/'):#e.g.: dt.generate_crystal_2d_graph(encodedgraphsavepath='./original_encoded_sites/',encodedlatticesavepath='./original_encoded_lattice/',crystal_2d_graph_path='./original_crystal_2d_graphs/')
	if not os.path.exists(crystal_2d_graph_path):
		os.makedirs(crystal_2d_graph_path)
	filename=os.listdir(encodedlatticesavepath)
	for eachnpyfile in filename:
		if eachnpyfile.endswith('.npy'):
			encodeddirectory=encodedlatticesavepath+eachnpyfile
			crystal_2d_graph=np.zeros([6,200])
			encoded_lattice=np.load(encodeddirectory)
			crystal_2d_graph[0,:]=encoded_lattice.reshape(200)
			encodeddirectory=encodedgraphsavepath+eachnpyfile
			for i in range(1,3):
				encoded_sites=np.load(encodeddirectory)[:,i-1]
				crystal_2d_graph[i,:]=encoded_sites.reshape(200)
			savefilename=crystal_2d_graph_path+eachnpyfile
			np.save(savefilename,crystal_2d_graph)
			for i in range(200):
				if crystal_2d_graph[0,i]!=encoded_lattice[i]:
					print(crystal_2d_graph[0,i],encoded_lattice[i])
					exit()

def change_lattice_in_crystal_2d_graph(previous_crystal_2d_graph_path='./crystal_2d_graphs/',encodedlatticesavepath='./encoded_lattice/',crystal_2d_graph_path='./crystal_2d_graphs/'):
	if not os.path.exists(crystal_2d_graph_path):
		os.makedirs(crystal_2d_graph_path)
	filename=os.listdir(encodedlatticesavepath)
	for eachnpyfile in filename:
		if eachnpyfile.endswith('.npy'):
			encodeddirectory=encodedlatticesavepath+eachnpyfile
			crystal_2d_graph=np.load(previous_crystal_2d_graph_path+eachnpyfile)#np.zeros([6,200])
			encoded_lattice=np.load(encodeddirectory)
			crystal_2d_graph[0,:]=encoded_lattice.reshape(200)
			savefilename=crystal_2d_graph_path+eachnpyfile
			np.save(savefilename,crystal_2d_graph)
			for i in range(200):
				if crystal_2d_graph[0,i]!=encoded_lattice[i]:
					print(crystal_2d_graph[0,i],encoded_lattice[i])
					exit()

def get_element_by_index(index):
    for ele_index, symbol in elements_with_index:
        if ele_index == index:
            return symbol
    return None  # Return None if the index is not found

import os
import pandas as pd

def filter_csv_by_existing_files(csv_file_path, folder_path, filename_column, output_csv_file):
    """
    This function filters rows from a CSV where the filename in a specified column
    exists in a given folder and saves the filtered rows to a new CSV file.

    Parameters:
    - csv_file_path: str, path to the original CSV file
    - folder_path: str, path to the folder where files are located
    - filename_column: str, column name in the CSV that contains filenames (without extensions)
    - output_csv_file: str, path to save the new filtered CSV
    """
    # Load the original CSV
    df = pd.read_csv(csv_file_path)

    # List to store the rows where the file exists
    rows_to_keep = []

    # Iterate through the rows of the DataFrame
    for index, row in df.iterrows():
        # Construct the full file path for each entry in the column
        filename = row[filename_column] + '.npy'
         # Add extension if necessary
        file_path = folder_path + '/' + filename
        print(file_path)
        
        # Check if the file exists in the folder
        if os.path.exists(file_path):
            print('appended ', file_path)
            rows_to_keep.append(row)  # If file exists, store the row

    # Create a new DataFrame from the filtered rows
    filtered_df = pd.DataFrame(rows_to_keep)

    # Save the new DataFrame to a new CSV file
    filtered_df.to_csv(output_csv_file, index=False)

    print(f'Filtered CSV saved to: {output_csv_file}')



def detect_peaks(image):
	#print(image)
	neighborhood = generate_binary_structure(3, 2)
	local_max = (maximum_filter(image, footprint = neighborhood, mode = "wrap") == image)
	#print(local_max)
	background = (image < 0.01)
	eroded_background = binary_erosion(background, structure = neighborhood, border_value = 1)
	detected_peaks = np.logical_and(local_max, np.logical_not(eroded_background))
	#detected_peaks = np.logical_not(eroded_background)
	
	return detected_peaks
'''
def reconstruction(image,ele):
	# image should have dimension of (N,N,N)
	image0 = gaussian_filter(image,sigma=0.15)
	peaks = detect_peaks(image0)

	recon_mat = Atoms(cell=15*np.identity(3),pbc=[1,1,1])
	(peak_x,peak_y,peak_z) = np.where(peaks==1.0)
	for px,py,pz in zip(peak_x,peak_y,peak_z):
		if np.sum(image[px-3:px+4,py-3:py+4,pz-3:pz+4] > 0) >= 0:
			recon_mat.append(Atom(ele,(px/64.0,py/64.0,pz/64.0)))

	pos = recon_mat.get_positions()
	recon_mat.set_scaled_positions(pos)

	return recon_mat

'''
def reconstruction(image, ele_index):
    # image should have dimension of (N,N,N)
    image0 = gaussian_filter(image, sigma=0.15)
    peaks = detect_peaks(image)
    #print(peaks)

    ele = get_element_by_index(ele_index)
    if ele is None:
        raise ValueError(f"Element with index {ele_index} not found")

    recon_mat = Atoms(cell=15*np.identity(3), pbc=[1, 1, 1])
    (peak_x, peak_y, peak_z) = np.where(peaks ==1.0)
    print('Total Peaks along x y z: ')
    print(len(peak_x), len(peak_y), len(peak_z))
    #if(len(peak_x)<=100 and len(peak_y)<=100 and len(peak_z)<=100):
    for px, py, pz in zip(peak_x, peak_y, peak_z):
            if np.sum(image[px-3:px+4, py-3:py+4, pz-3:pz+4] > 0) >= 0:
               recon_mat.append(Atom(ele, (px/64.0, py/64.0, pz/64.0)))  #was 64 instead of 32
               print('Positions: ', ele,px/64.0, py/64.0, pz/64.0)
            #print(np.sum(image[px-1:px+2, py-1:py+2, pz-1:pz+2] >=0))

    pos = recon_mat.get_positions()
    recon_mat.set_scaled_positions(pos)

    return recon_mat
	
from collections import Counter
import shutil

def save_cif_file(atoms, pretty_formula, file_path, save_dir="DATA"):
    """
    Helper function to save CIF files with the correct formula based on atomic symbols.
    """
    # Get symbols of atoms in the structure
    symbols = atoms.get_chemical_symbols()

    # Count occurrences of each symbol
    symbol_counts = Counter(symbols)

    # Construct the formula like Ge4Re2
    formula_parts = []
    for symbol, count in symbol_counts.items():
        if count > 1:
            formula_parts.append(f"{symbol}{count}")
        else:
            formula_parts.append(symbol)

    # Join the parts to form the final formula (e.g., Ge4Re2)
    new_file_name = ''.join(formula_parts) + ".cif"

    # Create the full path for saving the file
    save_path = os.path.join(save_dir, new_file_name)

    # Save the file (assuming some file-saving function like write_cif)
    shutil.copy(file_path, save_path)
    
    print(f"Saved {file_path} as {new_file_name} in {save_dir} folder")

def process_cif_files(text_directory, cif_path, data_path, output_excel):
    """
    Open CIF files, check if the extracted symbol matches the file name (pretty_formula),
    and save valid files to the 'DATA' folder. If the stoichiometric sum exceeds 20, skip saving the file.
    Additionally, create an Excel file with details of the valid CIF files saved.
    """

    # Read the CSV that contains the pretty_formula
    df = pd.read_csv(text_directory)

    # Create the output directory if it doesn't exist
    if not os.path.exists(data_path):
        os.makedirs(data_path)

    # Prepare a list to hold data for the Excel file
    valid_crystals = []

    # Helper function to construct formula while keeping original order
    def construct_ordered_formula(symbols):
        symbol_counts = Counter(symbols)
        formula_parts = []
        # Maintain the original order of appearance in the CIF file
        for symbol in symbols:
            if symbol_counts[symbol] > 0:
                count = symbol_counts[symbol]
                if count > 1:
                    formula_parts.append(f"{symbol}{count}")
                else:
                    formula_parts.append(symbol)
                # Remove the symbol from counter after processing
                symbol_counts[symbol] = 0
        return ''.join(formula_parts)

    # Loop through each CIF file from the DataFrame
    for index, row in df.iterrows():
        pretty_formula = row['pretty_formula']
        cif_filename = os.path.join(cif_path, f"{pretty_formula}.cif")

        if not os.path.exists(cif_filename):
            print(f"CIF file {cif_filename} not found.")
            continue

        try:
            # Load the CIF file using ASE's read function
            atoms = read(cif_filename)

            # Extract the atomic symbols and construct the ordered formula
            extracted_symbols = atoms.get_chemical_symbols()
            extracted_formula = construct_ordered_formula(extracted_symbols)

            # Calculate stoichiometric sum (total number of atoms)
            stoichiometric_sum = len(extracted_symbols)

            # 1st check: If extracted formula matches the pretty_formula, save it with the same name
            if extracted_formula == pretty_formula:
                shutil.copy(cif_filename, os.path.join(data_path, f"{pretty_formula}.cif"))
                print(f"Saved {cif_filename} as {pretty_formula}.cif in DATA folder")
                valid_crystals.append([pretty_formula, extracted_formula, stoichiometric_sum])

            # 2nd check: If stoichiometric sum is less than or equal to 20, save with extracted formula name
            elif stoichiometric_sum <= 20:
                shutil.copy(cif_filename, os.path.join(data_path, f"{extracted_formula}.cif"))
                print(f"Saved {cif_filename} as {extracted_formula}.cif in DATA folder (symbol mismatch but valid stoichiometry)")
                valid_crystals.append([pretty_formula, extracted_formula, stoichiometric_sum])

            # Skip if neither condition is met
            else:
                print(f"Skipping {cif_filename} (symbol mismatch and stoichiometric sum > 20)")

        except Exception as e:
            print(f"Error processing {cif_filename}: {e}")

    # Create a DataFrame from the valid crystals list
    valid_crystals_df = pd.DataFrame(valid_crystals, columns=["Pretty Formula (CSV)", "Extracted Formula (CIF)", "Stoichiometric Sum"])

    # Save the DataFrame to an Excel file
    valid_crystals_df.to_excel(output_excel, index=False)
    print(f"Excel file saved at {output_excel}")

def process_cif_files2(text_directory, cif_path, data_path, output_excel):
    """
    Open CIF files, check if the extracted symbol matches the file name (pretty_formula),
    and save valid files to the 'DATA' folder. If the stoichiometric sum exceeds 20, skip saving the file.
    Additionally, create an Excel file with details of the valid CIF files saved.
    """

    # Read the CSV that contains the pretty_formula
    df = pd.read_csv(text_directory)

    # Create the output directory if it doesn't exist
    if not os.path.exists(data_path):
        os.makedirs(data_path)

    # Prepare a list to hold data for the Excel file
    valid_crystals = []

    # Helper function to convert atomic symbols to formula form (e.g., A2B3)
    def convert_to_formula(symbols): #it extracted GeGeGe instead of Ge3 -> heres the code for that
        symbol_counts = Counter(symbols)
        formula = ''.join(f'{symbol}{count if count > 1 else ""}' for symbol, count in sorted(symbol_counts.items()))
        return formula

    # Loop through each CIF file from the DataFrame
    for index, row in df.iterrows():
        pretty_formula = row['pretty_formula']
        cif_filename = os.path.join(cif_path, f"{pretty_formula}.cif")

        if not os.path.exists(cif_filename):
            print(f"CIF file {cif_filename} not found.")
            continue

        try:
            # Load the CIF file using ASE's read function
            atoms = read(cif_filename)
            print(cif_filename)
            # Extract the atomic symbols and convert them to formula form (e.g., Hf3N2 instead of HfHfHfNN)
            #extracted_symbols = atoms.get_chemical_symbols()
            extracted_symbols = atoms.get_positions()
            #extracted_formula = convert_to_formula(extracted_symbols)

            # Calculate stoichiometric sum (total number of atoms)
            stoichiometric_sum = len(extracted_symbols)

            # 1st check: If extracted formula matches the pretty_formula, save it with the same name
            # if extracted_formula == pretty_formula:
            #     shutil.copy(cif_filename, os.path.join(data_path, f"{pretty_formula}.cif"))
            #     print(f"Saved {cif_filename} as {pretty_formula}.cif in DATA folder")
            #     valid_crystals.append([pretty_formula, extracted_formula, stoichiometric_sum])

            # 2nd check: If stoichiometric sum is less than or equal to 20, save with extracted formula name
            if stoichiometric_sum <= 20:
                shutil.copy(cif_filename, os.path.join(data_path, f"{pretty_formula}.cif"))
                print(f"Saved {cif_filename} as {pretty_formula}.cif in DATA folder ( valid stoichiometry)")
                print(stoichiometric_sum)
                #valid_crystals.append([pretty_formula, extracted_formula, stoichiometric_sum])

            # Skip if neither condition is met
            else:
                print(f"Skipping {cif_filename} (stoichiometric sum > 20)")
                print(stoichiometric_sum)

        except Exception as e:
            print(f"Error processing {cif_filename}: {e}")

    # Create a DataFrame from the valid crystals list
    #valid_crystals_df = pd.DataFrame(valid_crystals, columns=["Pretty Formula (CSV)", "Extracted Formula (CIF)", "Stoichiometric Sum"])

    # Save the DataFrame to an Excel file
    #valid_crystals_df.to_excel(output_excel, index=False)
   # print(f"Excel file saved at {output_excel}")

def reconstruction_positions(image, ele_index): #returns positions instead of reconstruction matrix
    # image should have dimension of (N,N,N)
    image0 = gaussian_filter(image, sigma=0.15)
    peaks = detect_peaks(image)
    #print(peaks)

    ele = get_element_by_index(ele_index)
    if ele is None:
        raise ValueError(f"Element with index {ele_index} not found")
    positions = []
    (peak_x, peak_y, peak_z) = np.where(peaks ==1.0)
    print('Total Peaks along x y z: ')
    print(len(peak_x), len(peak_y), len(peak_z))
    #if(len(peak_x)<=100 and len(peak_y)<=100 and len(peak_z)<=100):
    for px, py, pz in zip(peak_x, peak_y, peak_z):
            if np.sum(image[px-3:px+4, py-3:py+4, pz-3:pz+4] > 0) >= 0:
               pos = [px * 15 / 64.0, py *15 / 64.0, pz * 15/ 64.0]
               positions.append(pos) 
               print('Positions: ', ele,px/64.0, py/64.0, pz/64.0)
            #print(np.sum(image[px-1:px+2, py-1:py+2, pz-1:pz+2] >=0))

    return positions

'''

def reconstruction(image, ele):
    # Apply Gaussian filter to smooth the image
    image0 = gaussian_filter(image, sigma=0.15)
    
    # Detect peaks and get their coordinates and values
    peak_coords, peak_values = detect_peaks(image0)
    
    # Create an Atoms object with a unit cell of 15x15x15
    recon_mat = Atoms(cell=15*np.identity(3), pbc=[1, 1, 1])
    
    for (px, py, pz), prob_value in zip(peak_coords, peak_values):
        # Check if the probability value is greater than a threshold
        if prob_value > 0:
            recon_mat.append(Atom(ele, (px / 32.0, py / 32.0, pz / 32.0)))
    
    # Convert positions to scaled positions
    pos = recon_mat.get_positions()
    recon_mat.set_scaled_positions(pos)
    
    return recon_mat
'''

def generated_sites(genenrated_decoded_path='./generated_decoded_sites/',generated_pre_path='./generated_sites/',element_list=['Fe','Co']):
	if not os.path.exists(generated_pre_path):
		os.makedirs(generated_pre_path)

	filename=os.listdir(genenrated_decoded_path)
	bad_reproduce=0
	ele = element_list
	for eachfile in filename:
		if eachfile.endswith('.npy'):  #how to load a pkl file and change it to npy
			filename=genenrated_decoded_path+eachfile
			img=np.load(filename)

			tmp_mat = []
			for idc in range(2):
				image = img[:,:,:,idc].reshape(64,64,64)
				tmp_mat.append(reconstruction(image,ele[idc]))
			for atom in tmp_mat[-1]:
				tmp_mat[0].append(atom)	

			try:
				write(generated_pre_path+eachfile[:-4]+'.vasp',tmp_mat[0])
			except:
				bad_reproduce=bad_reproduce+1
				continue
	print(bad_reproduce)
'''
def compute_length(axis_val, voxel_size = 17.0): #17.0
    """
    Estimate the lattice parameter along a given axis.

    Parameters:
    axis_val (numpy array): Voxel values along the axis.
    voxel_size (float): Size of a single voxel in physical units.

    Returns:
    float: Estimated lattice parameter in physical units.
    """
    non_zeros = axis_val[axis_val > 0]

    # Fit a Gaussian distribution to the voxel values
    from scipy.stats import norm
    params = norm.fit(non_zeros)

    # Use the standard deviation as an estimate of the lattice parameter
    lattice_param = params[1] * voxel_size

    return lattice_param
'''
# def compute_length(axis_val):
# 	non_zeros = axis_val[axis_val > 0]
# 	print(non_zeros)
# 	(a,) = np.where(axis_val == non_zeros.min())

# 	# distance from center in grid space
# 	N = np.abs(16 - a[0])
# 	print(N)
# 	# length of the unit vector
# 	r_fake = np.sqrt(-2*0.26**2*np.log(non_zeros.min())) #r_fake = N*(r/32)
# 	r = r_fake * 32.0 / float(N)
# 	return r
#compute length function with max indices in descending order - THIS VERSION WAS USED FOR VALIDATION IN VOXEL GENERATION - SUCCESSFUL
def compute_length(axis_val):
    non_zeros = axis_val[axis_val > 0]
    print(non_zeros)
    a = np.argsort(-axis_val)#[axis_val != 0]

    # distance from center in grid space
    N = np.abs(16 - a[1])
    #print(a[1])
    p=1
    #if N == 0:
    #    N = np.abs(16 - a[1])
    #    p=1

    # length of the unit vector
    r_fake = np.sqrt((-2*0.26**2*np.log(axis_val[a[p]])))
    #print(axis_val[a[p]])
    r = r_fake * 32.0 / float(N) #
    return r

#create a comparison function which compares the values directed from both the csvs
#compare the values within a range of errors -> max of x(0.1) deflection from original values
#if deflection is less than the chosen value, continue the process 
#else create a list of those compounds whose values differed
#search in cleaned_data csv for those and re-run the process for those compounds with first atom logic
'''

def compute_length(axis_val):
	non_zeros = axis_val[axis_val > 0]
	#print(non_zeros)
	(a,) = np.where(axis_val == non_zeros.max())

	# distance from center in grid space
	N = np.abs(15 - a[1])
	if N==0:
		N = 1
	#print(N)
	
	# length of the unit vector
	r_fake = np.sqrt((-2*0.26**2*np.log((non_zeros.max())))) #r_fake = N*(r/32)
	r = r_fake * 32.0 / float(N)
	return r

'''

import numpy as np
# def compute_angle(ri, rj, rij):
#     v1 = np.array([ri, 0, 0])
#     v2 = np.array([rj, rij, 0])
#     dot_product = np.dot(v1, v2)
#     magnitude_product = np.linalg.norm(v1) * np.linalg.norm(v2)
#     cos_theta = dot_product / magnitude_product
#     theta = np.arccos(cos_theta)
#     if rij > ri:
#         theta = 2 * np.pi - theta
#     return 360 - np.rad2deg(theta)

	
def compute_angle(ri,rj,rij):
	cos_theta = (ri**2 + rj**2 - rij**2)/ (2*ri*rj)
	#cos_theta = (ri**2 + rj**2 - rij**2)/ (2*ri*rj)
	print('cos_theta :', cos_theta)
	theta = math.acos(cos_theta) * 180/np.pi 
	#print('theta :', theta)# angle in deg.
	return 180-theta 


def get_atoms(inputfile,filetype):
	atoms = read(inputfile,format = filetype)
	return atoms
'''
def generated_lattice(genenrated_decoded_path='./generated_decoded_lattice/',generated_pre_path='./generated_lattice/'):
	if not os.path.exists(generated_pre_path):
		os.makedirs(generated_pre_path)

	filename=os.listdir(genenrated_decoded_path)
	bad_reproduce=0
	for eachfile in filename:
		if eachfile.endswith('.npy'):

			filename=genenrated_decoded_path+eachfile
			img=np.load(filename)

			a_axis = img[:,16,16]
			try:
				ra = compute_length(a_axis)
			except ValueError:
				bad_reproduce=bad_reproduce+1
				continue
			b_axis = img[16,:,16]
			try:
				rb = compute_length(b_axis)
			except ValueError:
				bad_reproduce=bad_reproduce+1
				continue
			c_axis = img[16,16,:]
			try:
				rc = compute_length(c_axis)
			except ValueError:
				bad_reproduce=bad_reproduce+1
				continue
			try:
				ab_axis = np.array([img[i,i,16] for i in range(32)]); rab = compute_length(ab_axis)
			except ValueError:
				bad_reproduce=bad_reproduce+1
				continue
			try:
				bc_axis = np.array([img[16,i,i] for i in range(32)]); rbc = compute_length(bc_axis)
			except ValueError:
				bad_reproduce=bad_reproduce+1
				continue
			try:
				ca_axis = np.array([img[i,16,i] for i in range(32)]); rca = compute_length(ca_axis)
			except ValueError:
				bad_reproduce=bad_reproduce+1
				continue

			try:
				alpha = compute_angle(rb,rc,rbc)
			except ValueError:
				bad_reproduce=bad_reproduce+1
				continue
			try:
				beta = compute_angle(rc,ra,rca)
			except ValueError:
				bad_reproduce=bad_reproduce+1
				continue
			try:
				gamma = compute_angle(ra,rb,rab)
			except ValueError:
				bad_reproduce=bad_reproduce+1
				continue

			try:
				atoms = Atoms(cell=[ra,rb,rc,alpha,beta,gamma],pbc=True)
			except AssertionError:
				bad_reproduce=bad_reproduce+1
				continue
			atoms.append(Atom('Cu',[0.5]*3))
			pos = atoms.get_positions()
			atoms.set_scaled_positions(pos)
			try:
				write(generated_pre_path+eachfile[:-4]+'.vasp',atoms)
			except RuntimeError:
				bad_reproduce=bad_reproduce+1
				continue
	print(bad_reproduce)
'''
def generated_lattice(generated_decoded_path='./generated_decoded_lattice/', generated_pre_path='./generated_lattice/'):
    if not os.path.exists(generated_pre_path):
        os.makedirs(generated_pre_path)

    filenames = os.listdir(generated_decoded_path)
    bad_reproduce = 0
    x = 31 #0to 31 0 to 15 #16 before

    y = 32 #32
    # Initialize a list to store data for the CSV file
    data = []

    for eachfile in filenames:
        if eachfile.endswith('.npy'):
            filename = os.path.join(generated_decoded_path, eachfile)
            img = np.load(filename)
            print(filename)
            #print(img)

            a_axis = img[:, x, x]
            print(a_axis[a_axis>0])
            try:
                ra = compute_length(a_axis) #here
                print('a :',ra)
                
            except ValueError:
                bad_reproduce += 1
                print("a not calculated")				
                continue

            b_axis = img[x, :, x]
            print(b_axis[b_axis>0])
            try:
                rb = compute_length(b_axis)
                print('b :',rb)
            except ValueError:
                bad_reproduce += 1
                print("b not calculated")
                continue

            c_axis = img[x, x, :]
            print(c_axis[c_axis>0])
            try:
                rc = compute_length(c_axis)
                print('c :',rc)
            except ValueError:
                bad_reproduce += 1
                print("c not calculated")
                continue

            try:
                ab_axis = np.array([img[i, i, x] for i in range(y)])
                rab = compute_length(ab_axis)
            except ValueError:
                bad_reproduce += 1
                print("ab not calculated")
                continue

            try:
                bc_axis = np.array([img[x, i, i] for i in range(y)])
                rbc = compute_length(bc_axis)
            except ValueError:
                bad_reproduce += 1
                print("bc not calculated")
                continue

            try:
                ca_axis = np.array([img[i, x, i] for i in range(y)])
                rca = compute_length(ca_axis)
            except ValueError:
                bad_reproduce += 1
                print("ca not calculated")
                continue

            try:
                alpha = compute_angle(rb, rc, rbc)
                print('alpha: ',alpha)
            except ValueError:
                bad_reproduce += 1
                print('alpha not calculated')
                continue

            try:
                
                beta = compute_angle(rc, ra, rca)
                print('beta :',beta)
                 #problem
            except ValueError:
                bad_reproduce += 1 
                print('beta not calculated')
                continue

            try:
                gamma = compute_angle(ra, rb, rab)
                print('gamma :',gamma)
            except ValueError:
                bad_reproduce += 1
                print('gamma not calculated')
                continue
            data.append([eachfile,ra, rb, rc, alpha, beta, gamma])

    # Create a DataFrame and save to CSV
    df = pd.DataFrame(data, columns=['Crystal','a', 'b', 'c', 'alpha', 'beta', 'gamma'])
    df.to_csv('lattice_parameters_for_generated_crystals_decoded_GIT.csv', index=False)

    print(f'Number of bad reproductions: {bad_reproduce}')

#####train test generate
def train_test_split(path='./3d_crystal_graphs/',split_ratio=0.2):
	filename=os.listdir(path)
	name_list=[]
	for eachnpyfile in filename:
		if eachnpyfile.endswith('.npy'):
			name_list.append(eachnpyfile[:-4])
	test_size=round(split_ratio*len(name_list))
	random.shuffle(name_list)
	train_name_list=name_list[test_size:]
	test_name_list=name_list[:test_size]
	return test_size,test_name_list,train_name_list

#####get batch list
def get_batch_name_list(train_name_list,batch_size=24):
	random.shuffle(train_name_list)
	batch_name_list=train_name_list[:batch_size]
	return batch_name_list

#####get batch lattice input
def generate_lattice_batch(batch_size,latticesavepath='./lattice/',name_list=['mp-1183837_Co3Ni']):
	batch_lattices=np.zeros([batch_size,32,32,32,1])
	for i in range(0,batch_size):
		batch_lattices[i,:,:,:,:]=read_lattice(latticesavepath,name_list[i]).reshape([1,32,32,32,1])
	return batch_lattices

#####get batch element input
def generate_graph_batch(batch_size,graphsavepath='./3d_crystal_graphs/',name_list=['mp-1183837_Co3Ni'],element=0):
	batch_three_d_graphs=np.zeros([batch_size,64,64,64,2])
	for i in range(0,batch_size):
		batch_three_d_graphs[i,:,:,:,:]=read_crystal_graph(graphsavepath,name_list[i]).reshape([1,64,64,64,2])
		batch_three_d_graphs=batch_three_d_graphs[:,:,:,:,element].reshape([1,64,64,64,1])
	return batch_three_d_graphs

#####get batch 2d graph input
def generate_2dgraph_batch(batch_size,graph2d_savepath='./crystal_2d_graphs/',name_list=['mp-1183837_Co3Ni']):
	batch_2d_graphs=np.zeros([batch_size,1,200,6])
	for i in range(0,batch_size):
		batch_2d_graphs[i,:,:,:]=read_crystal_graph(graph2d_savepath,name_list[i]).reshape([1,1,200,6])
	return batch_2d_graphs

#####get 3d graph
def read_crystal_graph(graphsavepath='./3d_crystal_graphs/',name='mp-1183837_Co3Ni'):
	filename=graphsavepath+name+'.npy'
	three_d_graph=np.load(filename)
	return three_d_graph

#####get lattice
def read_lattice(latticesavepath='./lattice/',name='mp-1183837_Co3Ni'):
	filename=latticesavepath+name+'.npy'
	lattice=np.load(filename)
	return lattice
