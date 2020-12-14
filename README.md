
1D Radial Structure-from-Motion
======
This is an extension of the incremental Structure-from-Motion framework [COLMAP](https://github.com/colmap/colmap) which allows for using the 1D Radial camera model. This is a cleaned up re-implementation of the original code used for the experiments in the paper

`Larsson et al., Calibration-Free Structure-from-Motion with Calibrated Radial Trifocal Tensors, ECCV 2020`

and there might be some minor differences in the results. If you have any problems or find any bugs, please create an issue.
The implementation should work with a mix of camera models (where some are 1D Radial cameras) but this has not been tested thoroughly.

Example datasets
-----
Two example datasets can be downloaded from [here](https://drive.google.com/drive/folders/180lCeaaIT9uNXZ4eUvimrU0RgbC4d7gy?usp=sharing). In addition to the images, we also provide the COLMAP databases if you want to avoid the image matching.

**Amsterdam Square** (72 images, Fisheye)

Example initialization images: `R0010149b.jpg,R0010150b.jpg,R0010151b.jpg,R0010146b.jpg,R0010144b.jpg`
<img src="https://user-images.githubusercontent.com/48490995/102066495-2f52b580-3dfa-11eb-8737-1bbc2938d44c.png" height="250">
<img src="https://user-images.githubusercontent.com/48490995/102066639-5a3d0980-3dfa-11eb-9bd8-20306259157c.gif" height="250">

**Grossmunster Church** (372 images, Fisheye (cropped))

Example initialization images: `DSC_2637.JPG,DSC_2638.JPG,DSC_2639.JPG,DSC_2640.JPG,DSC_2641.JPG`
<img src="https://user-images.githubusercontent.com/48490995/102066770-86588a80-3dfa-11eb-964c-837a04cae108.png" height="250">
<img src="https://user-images.githubusercontent.com/48490995/102066820-96706a00-3dfa-11eb-8898-b9d3b503f2b9.gif" height="250">


Getting started
-----
For building from source, you can follow the intructions for vanilla COLMAP, see the [documentation](https://colmap.github.io/install.html).

The implementation does not include an automatic way to select initialization images so these must be supplied by the user during runtime. It is also possible to provide an initial reconstruction and continue the incremental SfM from this.

Below we show a short demo example of the reconstruction process for the Amsterdam Square dataset.

```bash
DATASET_PATH=~/datasets/ricoh_dam_square
INIT_IMAGES=R0010149b.jpg,R0010150b.jpg,R0010151b.jpg,R0010146b.jpg,R0010144b.jpg
OUTPUT_PATH=$DATASET_PATH/output

# It is important to make sure the cameras have the correct model!
./radial_colmap feature_extractor                           \
        --database_path $DATASET_PATH/database_radial.db    \
        --image_path $DATASET_PATH/images                   \
        --ImageReader.camera_model=1D_RADIAL

# COLMAP runs two-view geometric verification (assuming no distortion)
# For highly distorted images (as in this demo) we instead fit multiple models
# which allow more correct matches to pass this step. You can also increase 
# the thresholds significantly to achieve similar results.
./radial_colmap exhaustive_matcher                          \
        --database_path $DATASET_PATH/database_radial.db    \
        --SiftMatching.multiple_models=1

mkdir $OUTPUT_PATH

# The first three images in INIT_IMAGES are assumed to have intersecting principal axes.
./radial_colmap radial_trifocal_initializer                 \
        --database_path $DATASET_PATH/database_radial.db    \
        --image_path $DATASET_PATH/images/                  \
        --init_images $INIT_IMAGES                          \
        --output_path $OUTPUT_PATH

# Load the initialized reconstruction in the GUI. You can now press the play button 
# to start the reconstruction process.
./radial_colmap gui                                         \
        --database_path $DATASET_PATH/database_radial.db    \
        --image_path $DATASET_PATH/images/                  \
        --import_path $OUTPUT_PATH
        
# Alternatively you can also run the directly mapper from the cli
./radial_colmap mapper                                      \
        --database_path $DATASET_PATH/database_radial.db    \
        --image_path $DATASET_PATH/images/                  \
        --input_path $OUTPUT_PATH                           \
        --output_path $OUTPUT_PATH

```

In this GUI you can show the cameras (the principal axes) by holding ALT and scrolling the mouse wheel. The camera forward translations are guestimated by assuming a single focal length for the entire image. Note that this is only used for visualization purposes and to normalize the scale of the reconstruction.


## Citing
If you are using the library for (scientific) publications, please cite:

    @inproceedings{larsson2020calibration,
        author={Viktor Larsson, Nicolas Zobernig, Kasim Taskin, Marc Pollefeys},
        title={Calibration-Free Structure-from-Motion with Calibrated Radial Trifocal Tensors},
        booktitle={European Conference on Computer Vision (ECCV)},
        year={2020},
    }

This work heavily builds on COLMAP for which you should cite:

    @inproceedings{schoenberger2016sfm,
        author={Sch\"{o}nberger, Johannes Lutz and Frahm, Jan-Michael},
        title={Structure-from-Motion Revisited},
        booktitle={Conference on Computer Vision and Pattern Recognition (CVPR)},
        year={2016},
    }

    @inproceedings{schoenberger2016mvs,
        author={Sch\"{o}nberger, Johannes Lutz and Zheng, Enliang and Pollefeys, Marc and Frahm, Jan-Michael},
        title={Pixelwise View Selection for Unstructured Multi-View Stereo},
        booktitle={European Conference on Computer Vision (ECCV)},
        year={2016},
    }

If you use the image retrieval / vocabulary tree engine, please also cite:

    @inproceedings{schoenberger2016vote,
        author={Sch\"{o}nberger, Johannes Lutz and Price, True and Sattler, Torsten and Frahm, Jan-Michael and Pollefeys, Marc},
        title={A Vote-and-Verify Strategy for Fast Spatial Verification in Image Retrieval},
        booktitle={Asian Conference on Computer Vision (ACCV)},
        year={2016},
    }

The latest source code for COLMAP is available at https://github.com/colmap/colmap. COLMAP
builds on top of existing works and when using specific algorithms within
COLMAP, please also cite the original authors, as specified in the source code.


