# extract ROI-based values from a 3D nifti file (e.g. PET) and apply a grey matter mask to the atlas
# required input files are full paths to 3D nifti files, all images must be in the same space (i.e. MNI) and resolution
# the function returns a data frame with 1 row and n columns corresponding to the number of ROIs in the atlas

require(neurobase)
require(oro.nifti)
require(plyr)
require(dplyr)
require(tidyr)

# define the function
extract_Schaefer_ROIs <- function(path_to_pet, path_to_atlas, path_to_mask){
  # load files
  atlas_readin = neurobase::readnii(path_to_atlas)
  pet_readin = neurobase::readnii(path_to_pet)
  mask_readin = neurobase::readnii(path_to_mask)
  atlas_readin_masked = atlas_readin * mask_readin
  
  # reshape 3D nifti files to data frames
  tmp.df = 
    data.frame(atlas = as.numeric(atlas_readin_masked),
               nifti = as.numeric(pet_readin)) %>% 
    subset(atlas !=0)
  
  
  
  # extract mean 
  tmp.summary <- 
    tmp.df %>% 
    group_by(atlas) %>% 
    summarise(mean = mean(nifti, na.rm = T))
  
  # summarize mean ROI values
  tmp.summary.df <- data.frame(tmp.summary)
  #tmp.summary.df$atlas <- paste0("ROI_", tmp.summary.df$atlas)
  
  tmp.summary.df.wide = spread(tmp.summary.df, key = atlas, value = mean)
  
  return(tmp.summary.df.wide)
  
}

library("optparse")

option_list = list(
  make_option(c("-m", "--mask"), type="character", default=NULL, 
              help="Path to brainmask", metavar="character"),
  make_option(c("-a", "--atlas"), type="character", default=NULL, 
              help="Path to atlas", metavar="character"),
  make_option(c("-p", "--PET"), type="character", default=NULL, 
              help="Path to PET", metavar="character"),
  make_option(c("-o", "--out"), type="character", default=NULL, 
              help="Path to save output file to", metavar="character")
  
); 

opt_parser = OptionParser(option_list=option_list);
opt = parse_args(opt_parser);

args = commandArgs(trailingOnly=TRUE)
mask = opt$mask
atlas = opt$atlas
subject_PET = opt$PET

PET_Schaefer_single_sub = extract_Schaefer_ROIs(path_to_pet = subject_PET, path_to_atlas = atlas, path_to_mask = mask)
write.table(PET_Schaefer_single_sub, file=opt$out, row.names=FALSE, col.names=TRUE, sep=',')
print(PET_Schaefer_single_sub)

