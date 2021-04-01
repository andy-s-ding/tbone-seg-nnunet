'''
deploy_registrations.py

python3 -u registration.py --side RT --template 153 --moving 144

'''
import os
import argparse
import sys
import subprocess
import glob


def parse_command_line(args):
    '''

    '''

    print('parsing command line')

    parser = argparse.ArgumentParser(description='Registration pipeline for image-image registration on the cluster')

    parser.add_argument('--base', 
                        action="store", 
                        type=str, 
                        default="/scratch/groups/rtaylor2/ANTs-registration"
                        )
    parser.add_argument('--dry',
                        action="store_true"
                       )
    
    args = vars(parser.parse_args())
    return args

    
def prepare():
    
    os.environ['PATH'] = ':'.join(('/software/apps/slurm/current/bin/', os.environ['PATH']))
    
    args = parse_command_line(sys.argv)
    
    base = args['base']
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
    
    for i in range(len(scan_id)): 
        for j in range(len(scan_id)):
            if i == j: 
                continue
            job_file = os.path.join(os.getcwd(), "registration_%s_%s.sh"%(scan_id[i],scan_id[j]))
            
            with open(job_file, 'w') as fh:
                fh.writelines("#!/bin/bash\n")
                fh.writelines("#SBATCH --job-name=%s_%s\n"%(scan_id[i],scan_id[j]))
                fh.writelines("#SBATCH --output=/home-1/alu27@jhu.edu/taylorlab/asm/slurm_files/%s_%s.o%%j\n"%(scan_id[i], scan_id[j]))
                fh.writelines("#SBATCH --error=/home-1/alu27@jhu.edu/taylorlab/asm/slurm_files/%s_%s.e%%j\n"%(scan_id[i], scan_id[j]))
                fh.writelines("#SBATCH --account=rtaylor2\n")
                fh.writelines("#SBATCH --partition=parallel\n")
                fh.writelines("#SBATCH --nodes=1\n")
                fh.writelines("#SBATCH --ntasks-per-node=24\n")
                fh.writelines("#SBATCH --time=24:00:00\n")
                fh.writelines("#SBATCH --qos=normal\n")
                fh.writelines("#SBATCH --mail-type=ALL\n")
                fh.writelines("#SBATCH --mail-user=alu27@jhu.edu\n")
                fh.writelines("module list\n")
                fh.writelines(". /software/apps/anaconda/5.2/python/3.7.4/etc/profile.d/conda.sh \n")
                fh.writelines("conda activate lab_base\n")
                fh.writelines("python3 -u registration.py --side RT --template %s --moving %s\n"%(scan_id[i],scan_id[j]))

            if not dry_run: 
                os.system("sbatch %s" %job_file)
        
    return


def execute():
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
    
    main_script = os.path.join(os.getcwd(), "all.sh")
    with open(main_script, 'w') as fh: 
        fh.writelines("#!/bin/bash\n")
            
        for i in range(len(scan_id)): 
            for j in range(len(scan_id)):
                if i == j: 
                    continue
                fh.writelines("sbatch registration_%s_%s.sh\n"%(scan_id[i], scan_id[j]))
                
    return


def double_check(): 
    main_script = os.path.join(os.getcwd(), "rerun.sh")
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

    jobs = {}
    outfiles = glob.glob(os.path.join(os.getcwd(), "slurm_files/*.o*"))
    for outfile in outfiles: 
        with open(outfile, 'r') as f: 
            for line in f.readlines():
                if "entering registration" in line: 
                    tokens = line.split()
                    fixed = tokens[-4]
                    moving = tokens[-1]
                    if (fixed, moving) in jobs.keys(): 
                        jobs[(fixed, moving)] += [outfile]
                    else: 
                        jobs[(fixed, moving)] = [outfile]
                    break
    
    print(len(jobs.keys()))
    
    rerun_pairs = []
    for (fixed, moving), outfiles in jobs.items():
        
        outfile = sorted([(int(outfile[-8:]), outfile) for outfile in outfiles], key=lambda x: x[0])[-1][1]
            
        error_file = outfile.replace("n.o", "n.e")
        with open(error_file, 'r') as ef: 
            for line in ef.readlines():
                if "** ERROR: NWAD:" in line: 
                    rerun_pairs.append((fixed, moving))
                    break

    rerun_pairs = list(set(rerun_pairs))
    print(rerun_pairs)
    print(len(rerun_pairs))

    with open(main_script, "w") as fh: 
        fh.writelines("#!/bin/bash\n")
        for i, j in rerun_pairs: 
            fh.writelines("sbatch registration_%s_%s.sh\n"%(i, j))
                        
    return

                            
if __name__ == "__main__":
    prepare()
    execute()
#     double_check()
