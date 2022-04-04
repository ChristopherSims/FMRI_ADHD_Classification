#!/bin/bash

argc=$#
argv=("$@")

echo "$argc ${argv[0]}"

TEMPLATE_1mm=/home/drcc/ADHD200/templates/nihpd_asym_04.5-18.5_t1w.nii
TEMPLATE_2mm=/home/drcc/ADHD200/templates/nihpd_asym_04.5-18.5_t1w_2mm_ss.nii
TEMPLATE_4mm=/home/drcc/ADHD200/templates/nihpd_asym_04.5-18.5_t1w_4mm.nii
TEMPLATE_CFG=/home/drcc/ADHD200/templates/T1_2_NIHPD_2mm.cnf
PNAS_TEMPLATES=/home/drcc/ADHD200/templates/PNAS_Smith09_rsn10_nihpd_4mm.nii.gz
TR=2
MIDSLICE=17
ACQ=altplus

for (( i=0; i<argc; i++ ))
do
    dir=${argv[$i]}
    # verify that the file exists, is appropriate, and parse out 
    # relevant information
    if [ -d ${dir} ]; then
        subjid=`basename ${dir}`
        echo "$dir parsed into $subjid"
    else 
        echo "$dir does not exist" 
        continue
    fi

    # get a list of sessions
    sessions=`ls -d */`
  
    for session in ${sessions}
    do
        sid=`basename $session` 
        echo "processing $sid"

        #make sure there is an anatomical
        if [ -f ${dir}/${session}/anat_1/mprage_noface.nii.gz ]; then
            echo "${dir}/${session}/anat_1/mprage_noface.nii.gz found! processing..."
	else
            echo "${dir}/${session}/anat_1/mprage_noface.nii.gz not found! skipping..."
            continue
        fi 
 
        t1_file=d${subjid}_${sid}_anat.nii.gz
        if [ ! -f $t1_file ]; then
            # de-obliqued T1
            3dcopy  ${dir}/${session}/anat_1/mprage_noface.nii.gz rm_${t1_file} 
            3drefit -deoblique rm_${t1_file} 
            3dresample -orient RPI -prefix ${t1_file} -inset rm_${t1_file}
        fi
    
        ss_t1_file=ss${t1_file}
        if [ ! -f $ss_t1_file ]; then
            # skullstrip image
            3dSkullStrip -orig_vol -input ${t1_file} -prefix ss${t1_file}
        fi

        warp_file=${subjid}_${sid}_template.nii.gz
        if [ ! -f ${warp_file} ]; then
            # use flirt to normalize T1 to template
            flirt -ref ${TEMPLATE_2mm} -in ${ss_t1_file} -omat rm_affine_transf.mat
            fnirt --in=${t1_file} --aff=rm_affine_transf.mat \
                  --cout=${subjid}_${sid}_template --config=${TEMPLATE_CFG}
        fi
    
        norm_t1_file=w${t1_file}
        if [ ! -f ${norm_t1_file} ]; then 
            applywarp --ref=${TEMPLATE_1mm} --in=${t1_file} \
                      --warp=${subjid}_${sid}_template --out=${norm_t1_file}
        fi
    
        norm_ss_t1_file=w${ss_t1_file}
        if [ ! -f ${norm_ss_t1_file} ]; then 
            applywarp --ref=${TEMPLATE_1mm} --in=${ss_t1_file} \
                      --warp=${subjid}_${sid}_template --out=${norm_ss_t1_file}
        fi

	## create csf and wm masks in template space
        csf_mask=w${ss_t1_file%%.nii.gz}_csf.nii.gz
        wm_mask=w${ss_t1_file%%.nii.gz}_wm.nii.gz
        gm_mask=sw${ss_t1_file%%.nii.gz}_gm.nii.gz
    
        if [ -f ${csf_mask} -a -f ${wm_mask} ]; then
            echo "${csf_mask} and ${wm_mask} already exist"
        else
            in_csf_file=rm_${ss_t1_file%%.nii.gz}_pve_0.nii.gz
    	    in_wm_file=rm_${ss_t1_file%%.nii.gz}_pve_2.nii.gz

            if [ ! -f ${in_csf_file} -o ! -f ${in_wm_file} ]; then 
                fast --channels=1 --type=1 --class=3 --out=rm_${ss_t1_file} ${ss_t1_file}
            fi 

            # create csf mask in template space    
            if [ ! -f ${csf_mask} ]; then
                applywarp --ref=${TEMPLATE_1mm} --in=${in_csf_file} \
                          --warp=${subjid}_${sid}_template --out=rm_w${in_csf_file##rm_}
                3dcalc -a rm_w${in_csf_file##rm_} -expr 'step(a-.99)' -prefix ${csf_mask} -datum short
            fi
    
            # create white matter mask in template space    
            if [ ! -f ${wm_mask} ]; then
                applywarp --ref=${TEMPLATE_1mm} --in=${in_wm_file} \
                          --warp=${subjid}_${sid}_template --out=rm_w${in_wm_file##rm_}
                3dcalc -a rm_w${in_wm_file##rm_} -expr 'step(a-.99)' -prefix ${wm_mask} -datum short
            fi
        fi

        # gray matter mask in norm space
        if [ ! -f ${gm_mask} ]; then
            in_gm_file=rm_${ss_t1_file%%.nii.gz}_pve_1.nii.gz
            if [ ! -f ${in_gm_file} ]; then 
                fast --channels=1 --type=1 --class=3 --out=rm_${ss_t1_file} ${ss_t1_file}
            fi 
    
            applywarp --ref=${TEMPLATE_1mm} --in=${in_gm_file} \
                      --warp=${subjid}_${sid}_template --out=${gm_mask##s}
            3dmerge -1blur_fwhm 6 -doall -prefix ${gm_mask} ${gm_mask##s}
        fi
    
        #get a list of the rest directories
        restdirs=`ls -d ${dir}/${session}/rest*/`

        for restdir in ${restdirs}
        do
            # make sure the func exits
            if [ -f ${restdir}/rest.nii.gz ]; then
                echo "${restdir}/rest.nii.gz found, processing ..."
            else
                echo "${restdir}/rest.nii.gz not found, skipping ..."
                continue
            fi

            rid=`basename ${restdir}`
            #----- BASIC fMRI preprocessing
            bold_file=${subjid}_${sid}_${rid}.nii.gz
            orig_bold=rm_nmrda${bold_file}
            mni_preproc_bold_data=snwmrda${bold_file}
            norm_filtered_denoised_bold=sfnwmrda${bold_file}
            epi_mean_template=mean_mrda${bold_file}
            mni_mask_file=mask_w${epi_mean_template}

            if [ -f ${mni_preproc_bold_data} -a -f ${norm_filtered_denoised_bold} ]; then
                echo "${mni_preproc_bold_data} and ${norm_filtered_denoised_bold} already exist"
            else

                # coregister all EPI data to the first image of REST1
                coreg_base=rm_da${bold_file}'[0]'

                # remove first 4 images        
                if [ ! -f rm_${bold_file} ]; then
                    3dcalc -prefix rm_${bold_file} -a ${restdir}/rest.nii.gz'[4..$]' -expr 'a'
                fi
                prev_step=rm_${bold_file}

                # time shift dataset
                if [ ! -f rm_a${prev_step##rm_} ]; then
                    3dTshift -TR ${TR}s -slice ${MIDSLICE} -tpattern ${ACQ} -prefix rm_a${prev_step##rm_} ${prev_step}
                fi
                prev_step=rm_a${prev_step##rm_}

                # deoblique dataset, and convert to RPI
                if [ ! -f rm_d${prev_step##rm_} ]; then
                     3drefit -deoblique ${prev_step}
                     3dresample -orient RPI -prefix rm_d${prev_step##rm_} -inset ${prev_step}
                fi
                prev_step=rm_d${prev_step##rm_}
    
    
                # motion correct data
                motion_file=rp_${bold_file%%.nii.gz}.1D
                if [ -f rm_r${prev_step##rm_} -a -f ${motion_file} ]; then
                    echo "Both rm_r${prev_step##rm_} and ${motion_file} exist"
                else
                    3dvolreg -Fourier -prefix rm_r${prev_step##rm_} -base ${coreg_base} \
                        -1Dfile rp_${bold_file%%.nii.gz}.1D ${prev_step}
                fi
                prev_step=rm_r${prev_step##rm_}
    
                # create a mask for the dataset
                mask_file=mask_${prev_step##rm_}
                if [ ! -f ${mask_file} ]; then
                    echo "3dAutomask -prefix mask_${prev_step##rm_} ${prev_step}"
                    3dAutomask -prefix mask_${prev_step##rm_} ${prev_step}
                fi
    
                # mask the dataset
                if [ ! -f rm_m${prev_step##rm_} ]; then
                    3dcalc -a ${prev_step} -b ${mask_file} -expr 'ispositive(b)*a' -prefix rm_m${prev_step##rm_}
                fi
                prev_step=rm_m${prev_step##rm_}
                 
                # create an average of this file to use for coregistering to T1 at later stage
                if [ ! -f ${epi_mean_template} ]; then
                    3dTstat -prefix ${epi_mean_template} ${prev_step}
                fi
     
                #----- Transfrom BOLD data to template space
                # if it does not already exist, calculate transform from coreg_base to ANAT
                epi_mni_xform=${subjid}_${sid}_${rid}_epi_2_template_4mm
    
                # calculate the EPI-template transform
                if [ -f ${epi_mni_xform}.nii.gz ]; then
                     echo "${epi_mni_xform} already exists"
                else
             
                    # register coreg_base to anatomical
                    flirt -ref ${ss_t1_file} -in ${epi_mean_template} -dof 7 -omat rm_${subjid}_${sid}_${rid}_epi_2_T1.mat
    
                    # copy mean template into T1 space for debugging
                    flirt -in ${epi_mean_template} -ref ${ss_t1_file} -out t1_${epi_mean_template} \
                        -init rm_${subjid}_${sid}_${rid}_epi_2_T1.mat \
                        -applyxfm 
    
                     # combine xforms
                     convertwarp --ref=${TEMPLATE_4mm} --warp1=${warp_file} \
                         --premat=rm_${subjid}_${sid}_${rid}_epi_2_T1.mat --out=${epi_mni_xform} --relout 
                fi

                # copy mean image into template space for debugging
                if [ -f w{$epi_mean_template} ]; then  
                     echo "w${epi_mean_template} already exists"
                else
                     applywarp --ref=${TEMPLATE_4mm} \
                        --in=${epi_mean_template} --warp=${epi_mni_xform} --rel \
                        --out=w${epi_mean_template}
                fi

                # create mask in template space 
                if [ -f ${mni_mask_file} ]; then  
                     echo "${mni_mask_file} already exists"
                else
                    3dAutomask -prefix ${mni_mask_file} w${epi_mean_template}
                fi

                # copy data into template space 
                if [ ! -f rm_w${prev_step##rm_} ]; then
                    applywarp --ref=${TEMPLATE_4mm} \
                              --in=${prev_step} --warp=${epi_mni_xform} --rel \
                              --out=rm_w${prev_step##rm_}
                fi
                prev_step=rm_w${prev_step##rm_}

                #----- Nuisance variable regression
        
                # if it hasn't been done yet, fractionize wm and csf mask into BOLD space
                csf_mask_frac=${csf_mask%%.nii.gz}_frac.nii.gz
                if [ -f ${csf_mask_frac} ]; then
                    echo "Fractionated CSF mask already exists"
                else
                    3dfractionize -template ${prev_step} -input ${csf_mask} -preserve -prefix ${csf_mask_frac} 
                fi
    
                wm_mask_frac=${wm_mask%%.nii.gz}_frac.nii.gz
                if [ -f ${wm_mask_frac} ]; then
                    echo "Fractionated WM mask already exists"
                else
                    3dfractionize -template ${prev_step} -input ${wm_mask} -preserve -prefix ${wm_mask_frac} 
                fi
    
                # extract wm and csf timecourses
                csf_tc_file=rm_csf_${prev_step##rm_}
                csf_tc_file=${csf_tc_file%%.nii.gz}.1D
                if [ ! -f ${csf_tc_file} ]; then
                    3dmaskave -q -mask ${csf_mask_frac} ${prev_step} > ${csf_tc_file}
                fi

                wm_tc_file=rm_wm_${prev_step##rm_}
                wm_tc_file=${wm_tc_file%%.nii.gz}.1D
                if [ ! -f ${wm_tc_file} ]; then
                    3dmaskave -q -mask ${wm_mask_frac} ${prev_step} > ${wm_tc_file}
                fi
    
                if [ ! -f rm_n${prev_step##rm_} ]; then
                    # perform nuisance variable regression
                    3dDeconvolve -polort A -num_stimts 8 \
                        -stim_file 1 ${motion_file}'[0]' -stim_base 1 -stim_label 1 roll \
                        -stim_file 2 ${motion_file}'[1]' -stim_base 2 -stim_label 2 pitch \
                        -stim_file 3 ${motion_file}'[2]' -stim_base 3 -stim_label 3 yaw \
                        -stim_file 4 ${motion_file}'[3]' -stim_base 4 -stim_label 4 dS \
                        -stim_file 5 ${motion_file}'[4]' -stim_base 5 -stim_label 5 dL \
                        -stim_file 6 ${motion_file}'[5]' -stim_base 6 -stim_label 6 dP \
                        -stim_file 7 ${csf_tc_file} -stim_base 7 -stim_label 7 csf \
                        -stim_file 8 ${wm_tc_file} -stim_base 8 -stim_label 8 wm \
                        -TR_1D ${TR}s -bucket rm_${subjid}_${sid}_${rid}_bucket -cbucket rm_${subjid}_${sid}_${rid}_cbucket \
                        -x1D rm_${subjid}_${sid}_${rid}_x1D.xmat.1D -input ${prev_step} -errts rm_n${prev_step##rm_}
                fi
                prev_step=rm_n${prev_step##rm_}
    
                ###  This is endpoint #1, slice timing corrected, motion corrected, masked, nuisance removed,
                #    transformed to template space and smoothed smooth data with 6mm FWHM
                if [ ! -f s${prev_step##rm_} ]; then
                    3dmerge -1blur_fwhm 6 -doall -prefix s${prev_step##rm_} ${prev_step}
                fi
    
                #----- Bandpass filter
                # make sure that the TR is correct
                3drefit -TR ${TR}s ${prev_step}

                filtered_file=sf${prev_step##rm_}
                if [ ! -f ${filtered_file} ]; then
                    if [ ! -f rm_f${prev_step##rm_} ]; then 
                        3dBandpass -prefix rm_f${prev_step##rm_} 0.009 0.08 ${prev_step}
                    fi
                    prev_step=rm_f${prev_step##rm_}
    
                    ###  This is endpoint #2, slice timing corrected, motion corrected, masked, nuisance removed,
                    # filtered, transformed to template space and smoothed smooth data with 6mm FWHM
                    if [ ! -f s${prev_step##rm_} ]; then 
                        3dmerge -1blur_fwhm 6 -doall -prefix s${prev_step##rm_} ${prev_step}
                    fi
                fi

            fi

            # calculate fALFF maps
            if [ -f falff_${bold_file} ]; then
                echo "already computed fALFF";
            else
                3dTstat -stdev -mask ${mni_mask_file} -prefix rm_falff_num_${bold_file} \
                    ${norm_filtered_denoised_bold}

                3dTstat -stdev -mask ${mni_mask_file} -prefix rm_falff_denom_${bold_file} \
                    ${mni_preproc_bold_data}

                3dcalc -prefix falff_${bold_file} -a ${mni_mask_file} -b rm_falff_num_${bold_file} \
                    -c rm_falff_denom_${bold_file} -expr '(1.0*bool(a))*((1.0*b)/(1.0*c))' -float
            fi

            if [ -f ${PNAS_TEMPLATES} ]
            then
                # extract TC for Smiths PNAS 10 RSNs in template space
                if [ -f ${mni_preproc_bold_data%%.nii.gz}_TCs.1D ]; then
                    echo " ${mni_preproc_bold_data%%.nii.gz}_TCs.1D already extracted timecourses in template space";
                else
                    fsl_glm -i ${mni_preproc_bold_data} -d ${PNAS_TEMPLATES} \
                            -o ${mni_preproc_bold_data%%.nii.gz}_TCs.1D
                fi 

                # calculate FC maps for each of the extracted TCs
                if [ -f fc_${mni_preproc_bold_data} ]; then
                    echo "fc_${mni_preproc_bold_data} already exists"
                else
                    3dTcorr1D -pearson -prefix fc_${mni_preproc_bold_data} -mask ${mni_mask_file} \
                        ${mni_preproc_bold_data} ${mni_preproc_bold_data%%.nii.gz}_TCs.1D
                fi
            fi
        done
    done
done

# delete intermediate files
