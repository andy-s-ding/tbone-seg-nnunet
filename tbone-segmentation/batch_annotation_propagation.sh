## batch evaluation
#!/bin/bsh

base="/Users/alex/Documents/segmentations/"
predictions="/Users/alex/Documents/predictions/"

# base="/scratch/groups/rtaylor2/ANTs-registration/segmentations"
# predictions="/scratch/groups/rtaylor2/ANTs-registration/predictions"

# prop annotations onto malleus
# python3 annotation_propagation.py --base /Users/alex/Documents/TaylorLab/asm --targets "RT 138 malleus mesh.vtk" --hardcoded_options malleus

# python3 annotation_propagation.py --base /Users/alex/Documents/TaylorLab/asm --targets "RT 142 Malleus mesh.vtk" --hardcoded_options malleus

# python3 annotation_propagation.py --base /Users/alex/Documents/TaylorLab/asm --targets "RT 143 Malleus mesh.vtk" --hardcoded_options malleus

# python3 annotation_propagation.py --base /Users/alex/Documents/TaylorLab/asm --targets "RT 144 malleus mesh.vtk" --hardcoded_options malleus

# python3 annotation_propagation.py --base /Users/alex/Documents/TaylorLab/asm --targets "RT 146 Malleus mesh.vtk" --hardcoded_options malleus

# python3 annotation_propagation.py --base /Users/alex/Documents/TaylorLab/asm --targets "RT 147 malleus mesh.vtk" --hardcoded_options malleus

# python3 annotation_propagation.py --base /Users/alex/Documents/TaylorLab/asm --targets "RT 150 malleus mesh.vtk" --hardcoded_options malleus

# python3 annotation_propagation.py --base /Users/alex/Documents/TaylorLab/asm --targets "RT 152 malleus mesh.vtk" --hardcoded_options malleus


# prop annotations onto chorda
# python3 annotation_propagation.py --base /Users/alex/Documents/TaylorLab/asm --targets "RT 138 chorda tympani mesh.vtk" --hardcoded_options chorda

# python3 annotation_propagation.py --base /Users/alex/Documents/TaylorLab/asm --targets "RT 142 Chorda Tympani mesh.vtk" --hardcoded_options chorda

# python3 annotation_propagation.py --base /Users/alex/Documents/TaylorLab/asm --targets "RT 143 Chorda Tympani mesh.vtk" --hardcoded_options chorda

# python3 annotation_propagation.py --base /Users/alex/Documents/TaylorLab/asm --targets "RT 144 chorda tympani mesh.vtk" --hardcoded_options chorda

# python3 annotation_propagation.py --base /Users/alex/Documents/TaylorLab/asm --targets "RT 146 Chorda Tympani mesh.vtk" --hardcoded_options chorda

# python3 annotation_propagation.py --base /Users/alex/Documents/TaylorLab/asm --targets "RT 147 chorda tympani mesh.vtk" --hardcoded_options chorda

# python3 annotation_propagation.py --base /Users/alex/Documents/TaylorLab/asm --targets "RT 150 chorda tympani mesh.vtk" --hardcoded_options chorda

# python3 annotation_propagation.py --base /Users/alex/Documents/TaylorLab/asm --targets "RT 152 chorda tympani mesh.vtk" --hardcoded_options chorda


# prop annotations onto facial 
# python3 annotation_propagation.py --base /Users/alex/Documents/TaylorLab/asm --targets "RT 138 facial nerve mesh.vtk" --hardcoded_options facial --decimation_factor .9

# python3 annotation_propagation.py --base /Users/alex/Documents/TaylorLab/asm --targets "RT 142 Facial Nerve mesh.vtk" --hardcoded_options facial --decimation_factor .9

# python3 annotation_propagation.py --base /Users/alex/Documents/TaylorLab/asm --targets "RT 143 Facial Nerve mesh.vtk" --hardcoded_options facial --decimation_factor .9

# python3 annotation_propagation.py --base /Users/alex/Documents/TaylorLab/asm --targets "RT 144 facial nerve mesh.vtk" --hardcoded_options facial --decimation_factor .9

# python3 annotation_propagation.py --base /Users/alex/Documents/TaylorLab/asm --targets "RT 146 Facial Nerve mesh.vtk" --hardcoded_options facial --decimation_factor .9

# python3 annotation_propagation.py --base /Users/alex/Documents/TaylorLab/asm --targets "RT 147 facial nerve mesh.vtk" --hardcoded_options facial --decimation_factor .9

# python3 annotation_propagation.py --base /Users/alex/Documents/TaylorLab/asm --targets "RT 150 facial nerve mesh.vtk" --hardcoded_options facial --decimation_factor .9

# python3 annotation_propagation.py --base /Users/alex/Documents/TaylorLab/asm --targets "RT 152 facial nerve mesh.vtk" --hardcoded_options facial --decimation_factor .9


# prop annotations onto EAC
# python3 annotation_propagation.py --base /Users/alex/Documents/TaylorLab/asm --targets "RT 138 facial nerve mesh.vtk" --hardcoded_options facial --decimation_factor .9

# python3 annotation_propagation.py --base /Users/alex/Documents/TaylorLab/asm --targets "RT 142 EAC mesh.vtk" --hardcoded_options EAC --decimation_factor .97

# python3 annotation_propagation.py --base /Users/alex/Documents/TaylorLab/asm --targets "RT 143 EAC mesh.vtk" --hardcoded_options EAC --decimation_factor .97

# python3 annotation_propagation.py --base /Users/alex/Documents/TaylorLab/asm --targets "RT 144 EAC mesh.vtk" --hardcoded_options EAC --decimation_factor .97

# python3 annotation_propagation.py --base /Users/alex/Documents/TaylorLab/asm --targets "RT 146 EAC mesh.vtk" --hardcoded_options EAC --decimation_factor .97

# python3 annotation_propagation.py --base /Users/alex/Documents/TaylorLab/asm --targets "RT 147 EAC mesh.vtk" --hardcoded_options EAC --decimation_factor .97

# python3 annotation_propagation.py --base /Users/alex/Documents/TaylorLab/asm --targets "RT 150 EAC mesh.vtk" --hardcoded_options EAC --decimation_factor .97

# python3 annotation_propagation.py --base /Users/alex/Documents/TaylorLab/asm --targets "RT 152 EAC mesh.vtk" --hardcoded_options EAC --decimation_factor .97


# prop annotations onto incus
python3 annotation_propagation.py --base /Users/alex/Documents/TaylorLab/asm --targets "RT 138 incus mesh.vtk" --hardcoded_options incus

python3 annotation_propagation.py --base /Users/alex/Documents/TaylorLab/asm --targets "RT 142 Incus mesh.vtk" --hardcoded_options incus 

python3 annotation_propagation.py --base /Users/alex/Documents/TaylorLab/asm --targets "RT 143 Incus mesh.vtk" --hardcoded_options incus 

python3 annotation_propagation.py --base /Users/alex/Documents/TaylorLab/asm --targets "RT 144 incus mesh.vtk" --hardcoded_options incus 

python3 annotation_propagation.py --base /Users/alex/Documents/TaylorLab/asm --targets "RT 146 Incus mesh.vtk" --hardcoded_options incus 

python3 annotation_propagation.py --base /Users/alex/Documents/TaylorLab/asm --targets "RT 147 incus mesh.vtk" --hardcoded_options incus 

python3 annotation_propagation.py --base /Users/alex/Documents/TaylorLab/asm --targets "RT 150 incus mesh.vtk" --hardcoded_options incus 

python3 annotation_propagation.py --base /Users/alex/Documents/TaylorLab/asm --targets "RT 152 incus mesh.vtk" --hardcoded_options incus 
