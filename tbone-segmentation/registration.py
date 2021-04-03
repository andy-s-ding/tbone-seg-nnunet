'''
registration.py

Script for using ANTSpy to do big registration 

'''
import numpy as np
import ants
import os 
import sys 
import argparse 
# from argparse import RawTextHelpFormatter
import nibabel as nib
import shutil
import psutil
import gc

def parse_command_line(args):
    '''

    '''

    print('parsing command line')

    parser = argparse.ArgumentParser(description='Registration pipeline for image-image registration')
    parser.add_argument('--template', 
                        action="store", 
                        type=str, 
                        default="142"
                        )
    parser.add_argument('--side', 
                        action="store", 
                        type=str, 
                        default="RT"
                        )
    parser.add_argument('--base', 
                        action="store", 
                        type=str, 
                        default="/scratch/groups/rtaylor2/ANTs-registration"
                        )
    parser.add_argument('--moving',
                        action="store",
                        type=str,
                        help="specify which scan should be the moving scan. If a scan is specified, only the registration of the specified moving scan onto the template scan will be performed. If the moving scan is not specified, the entire list of scans will be registered onto the template." 
                       )
    parser.add_argument('--dry',
                        action="store_true"
                       )
    
    args = vars(parser.parse_args())
    return args


def execute_registration(template, scan, base, side, save_dir, dry=False): 
    '''
    
    '''
    
    print('---'*10)
    print('entering registration w/ fixed %s and moving %s'%(template, scan))
    
    fixed_path = os.path.join(base, 'images', side + ' ' + template + '.nrrd')
    moving_path = os.path.join(base, 'images', side + ' ' + scan + '.nrrd')
      
    similarity_path = os.path.join(save_dir, 'similarity/%s %s %s similarity.mat'%(side, template, scan))
    deform_path = os.path.join(save_dir, 'new_deform/%s %s %s inverse.nii.gz'%(side, template, scan))
    
    # affine_path = os.path.join(save_dir, '%s %s %s affine.mat'%(side, template, scan))
    # deform_path = os.path.join(save_dir, '%s %s %s inverse.nii.gz'%(side, template, scan))
  
    if dry:
        print(fixed_path, moving_path, deform_path, similarity_path)
#         print(fixed_path, moving_path, deform_path, affine_path)
    
    if not dry: 
        fixed = ants.image_read(fixed_path)
        moving = ants.image_read(moving_path)

        fixed = ants.resample_image(fixed, (100, 100, 100), 1, 0)
        moving = ants.resample_image(moving, (100, 100, 100), 1, 0)
        
        transform_similarity = ants.registration(fixed=fixed , moving=moving, type_of_transform='Similarity', verbose=True)

        print("similarity transform complete")

        transform_syn = ants.registration(fixed=fixed, moving=moving, type_of_transform="SyNOnly", verbose=True)
    
        print("syn transform complete")

        # shutil.move(transform_similarity['invtransforms'][0], similarity_path)
        # # shutil.move(transform_syn['invtransforms'][0], affine_path)
        # shutil.move(transform_syn['invtransforms'][0], deform_path)

        print(transform_similarity['invtransforms'])
        print(transform_syn['invtransforms'])
        # del transform_similarity
        # del transform_syn
        gc.collect()
    
    return 


def main():
    
    args = parse_command_line(sys.argv)
    
    side = args['side']
    base = args['base']
    template = args['template']
    dry_run = args['dry']
    
    images = os.path.join(base, 'images')
    save_dir = os.path.join(base, 'transforms')
    
    scan_id = ['140', 
               '142', 
               '143', 
               '144', 
               '145', 
               '146', 
               '147', 
               '152', 
               '153'
              ]
    
    
    if args['moving'] is not None: 
        if args['moving'] not in scan_id: 
            print('incorrectly specified moving scan')
            return
        scan_id = [args['moving']]

    for scan in scan_id:
        if template in scan: 
            continue
            
        execute_registration(template, scan, base, side, save_dir, dry=dry_run)
        
        process = psutil.Process(os.getpid())
        print(process.memory_info().rss)
        
    return


if __name__ == "__main__":
    main()
