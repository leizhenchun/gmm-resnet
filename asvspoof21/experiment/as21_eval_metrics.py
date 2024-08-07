########################################
#        Zhenchun Lei
#  zhenchun.lei@hotmail.com
########################################

import logging
import os.path
import os.path
import sys

import numpy as np
import pandas

from asvspoof21.util.as_logging import logger_init_basic


def obtain_asv_error_rates(tar_asv, non_asv, spoof_asv, asv_threshold):
    # False alarm and miss rates for ASV
    Pfa_asv = sum(non_asv >= asv_threshold) / non_asv.size
    Pmiss_asv = sum(tar_asv < asv_threshold) / tar_asv.size

    # Rate of rejecting spoofs in ASV
    if spoof_asv.size == 0:
        Pmiss_spoof_asv = None
        Pfa_spoof_asv = None
    else:
        Pmiss_spoof_asv = np.sum(spoof_asv < asv_threshold) / spoof_asv.size
        Pfa_spoof_asv = np.sum(spoof_asv >= asv_threshold) / spoof_asv.size

    return Pfa_asv, Pmiss_asv, Pmiss_spoof_asv, Pfa_spoof_asv


def compute_det_curve(target_scores, nontarget_scores):
    n_scores = target_scores.size + nontarget_scores.size
    all_scores = np.concatenate((target_scores, nontarget_scores))
    labels = np.concatenate((np.ones(target_scores.size), np.zeros(nontarget_scores.size)))

    # Sort labels based on scores
    indices = np.argsort(all_scores, kind='mergesort')
    labels = labels[indices]

    # Compute false rejection and false acceptance rates
    tar_trial_sums = np.cumsum(labels)
    nontarget_trial_sums = nontarget_scores.size - (np.arange(1, n_scores + 1) - tar_trial_sums)

    frr = np.concatenate((np.atleast_1d(0), tar_trial_sums / target_scores.size))  # false rejection rates
    far = np.concatenate((np.atleast_1d(1), nontarget_trial_sums / nontarget_scores.size))  # false acceptance rates
    thresholds = np.concatenate(
        (np.atleast_1d(all_scores[indices[0]] - 0.001), all_scores[indices]))  # Thresholds are the sorted scores

    return frr, far, thresholds


def compute_eer(target_scores, nontarget_scores):
    """ Returns equal error rate (EER) and the corresponding threshold. """
    frr, far, thresholds = compute_det_curve(target_scores, nontarget_scores)
    abs_diffs = np.abs(frr - far)
    min_index = np.argmin(abs_diffs)
    eer = np.mean((frr[min_index], far[min_index]))
    return eer, thresholds[min_index]


def compute_tDCF(bonafide_score_cm, spoof_score_cm, Pfa_asv, Pmiss_asv, Pfa_spoof_asv, cost_model, print_cost):
    """
    Compute Tandem Detection Cost Function (t-DCF) [1] for a fixed ASV system.
    In brief, t-DCF returns a detection cost of a cascaded system of this form,

      Speech waveform -> [CM] -> [ASV] -> decision

    where CM stands for countermeasure and ASV for automatic speaker
    verification. The CM is therefore used as a 'gate' to decided whether or
    not the input speech sample should be passed onwards to the ASV system.
    Generally, both CM and ASV can do detection errors. Not all those errors
    are necessarily equally cost, and not all types of users are necessarily
    equally likely. The tandem t-DCF gives a principled with to compare
    different spoofing countermeasures under a detection cost function
    framework that takes that information into account.

    INPUTS:

      bonafide_score_cm   A vector of POSITIVE CLASS (bona fide or human)
                          detection scores obtained by executing a spoofing
                          countermeasure (CM) on some positive evaluation trials.
                          trial represents a bona fide case.
      spoof_score_cm      A vector of NEGATIVE CLASS (spoofing attack)
                          detection scores obtained by executing a spoofing
                          CM on some negative evaluation trials.
      Pfa_asv             False alarm (false acceptance) rate of the ASV
                          system that is evaluated in tandem with the CM.
                          Assumed to be in fractions, not percentages.
      Pmiss_asv           Miss (false rejection) rate of the ASV system that
                          is evaluated in tandem with the spoofing CM.
                          Assumed to be in fractions, not percentages.
      Pmiss_spoof_asv     Miss rate of spoof samples of the ASV system that
                          is evaluated in tandem with the spoofing CM. That
                          is, the fraction of spoof samples that were
                          rejected by the ASV system.
      cost_model          A struct that contains the parameters of t-DCF,
                          with the following fields.

                          Ptar        Prior probability of target speaker.
                          Pnon        Prior probability of nontarget speaker (zero-effort impostor)
                          Psoof       Prior probability of spoofing attack.
                          Cmiss       Cost of tandem system falsely rejecting target speaker.
                          Cfa         Cost of tandem system falsely accepting nontarget speaker.
                          Cfa_spoof   Cost of tandem system falsely accepting spoof.

      print_cost          Print a summary of the cost parameters and the
                          implied t-DCF cost function?

    OUTPUTS:

      tDCF_norm           Normalized t-DCF curve across the different CM
                          system operating points; see [2] for more details.
                          Normalized t-DCF > 1 indicates a useless
                          countermeasure (as the tandem system would do
                          better without it). min(tDCF_norm) will be the
                          minimum t-DCF used in ASVspoof 2019 [2].
      CM_thresholds       Vector of same size as tDCF_norm corresponding to
                          the CM threshold (operating point).

    NOTE:
    o     In relative terms, higher detection scores values are assumed to
          indicate stronger support for the bona fide hypothesis.
    o     You should provide real-valued soft scores, NOT hard decisions. The
          recommendation is that the scores are log-likelihood ratios (LLRs)
          from a bonafide-vs-spoof hypothesis based on some statistical model.
          This, however, is NOT required. The scores can have arbitrary range
          and scaling.
    o     Pfa_asv, Pmiss_asv, Pmiss_spoof_asv are in fractions, not percentages.

    References:

      [1] T. Kinnunen, H. Delgado, N. Evans,K.-A. Lee, V. Vestman, 
          A. Nautsch, M. Todisco, X. Wang, M. Sahidullah, J. Yamagishi, 
          and D.-A. Reynolds, "Tandem Assessment of Spoofing Countermeasures
          and Automatic Speaker Verification: Fundamentals," IEEE/ACM Transaction on
          Audio, Speech and Language Processing (TASLP).

      [2] ASVspoof 2019 challenge evaluation plan
          https://www.asvspoof.org/asvspoof2019/asvspoof2019_evaluation_plan.pdf
    """

    # Sanity check of cost parameters
    if cost_model['Cfa'] < 0 or cost_model['Cmiss'] < 0 or \
            cost_model['Cfa'] < 0 or cost_model['Cmiss'] < 0:
        print('WARNING: Usually the cost values should be positive!')

    if cost_model['Ptar'] < 0 or cost_model['Pnon'] < 0 or cost_model['Pspoof'] < 0 or \
            np.abs(cost_model['Ptar'] + cost_model['Pnon'] + cost_model['Pspoof'] - 1) > 1e-10:
        sys.exit('ERROR: Your prior probabilities should be positive and sum up to one.')

    # Unless we evaluate worst-case model, we need to have some spoof tests against asv
    if Pfa_spoof_asv is None:
        sys.exit('ERROR: you should provide false alarm rate of spoof tests against your ASV system.')

    # Sanity check of scores
    combined_scores = np.concatenate((bonafide_score_cm, spoof_score_cm))
    if np.isnan(combined_scores).any() or np.isinf(combined_scores).any():
        sys.exit('ERROR: Your scores contain nan or inf.')

    # Sanity check that inputs are scores and not decisions
    n_uniq = np.unique(combined_scores).size
    if n_uniq < 3:
        sys.exit('ERROR: You should provide soft CM scores - not binary decisions')

    # Obtain miss and false alarm rates of CM
    Pmiss_cm, Pfa_cm, CM_thresholds = compute_det_curve(bonafide_score_cm, spoof_score_cm)

    # Constants - see ASVspoof 2019 evaluation plan

    C0 = cost_model['Ptar'] * cost_model['Cmiss'] * Pmiss_asv + cost_model['Pnon'] * cost_model['Cfa'] * Pfa_asv
    C1 = cost_model['Ptar'] * cost_model['Cmiss'] - (
            cost_model['Ptar'] * cost_model['Cmiss'] * Pmiss_asv + cost_model['Pnon'] * cost_model['Cfa'] * Pfa_asv)
    C2 = cost_model['Pspoof'] * cost_model['Cfa_spoof'] * Pfa_spoof_asv;

    # Sanity check of the weights
    if C0 < 0 or C1 < 0 or C2 < 0:
        sys.exit(
            'You should never see this error but I cannot evalute tDCF with negative weights - please check whether your ASV error rates are correctly computed?')

    # Obtain t-DCF curve for all thresholds
    tDCF = C0 + C1 * Pmiss_cm + C2 * Pfa_cm

    # Obtain default t-DCF
    tDCF_default = C0 + np.minimum(C1, C2)

    # Normalized t-DCF
    tDCF_norm = tDCF / tDCF_default

    # Everything should be fine if reaching here.
    if print_cost:
        print(
            't-DCF evaluation from [Nbona={}, Nspoof={}] trials\n'.format(bonafide_score_cm.size, spoof_score_cm.size))
        print('t-DCF MODEL')
        print('   Ptar         = {:8.5f} (Prior probability of target user)'.format(cost_model['Ptar']))
        print('   Pnon         = {:8.5f} (Prior probability of nontarget user)'.format(cost_model['Pnon']))
        print('   Pspoof       = {:8.5f} (Prior probability of spoofing attack)'.format(cost_model['Pspoof']))
        print(
            '   Cfa          = {:8.5f} (Cost of tandem system falsely accepting a nontarget)'.format(cost_model['Cfa']))
        print('   Cmiss        = {:8.5f} (Cost of tandem system falsely rejecting target speaker)'.format(
            cost_model['Cmiss']))
        print(
            '   Cfa_spoof    = {:8.5f} (Cost of tandem sysmte falsely accepting spoof)'.format(cost_model['Cfa_spoof']))
        print('\n   Implied normalized t-DCF function (depends on t-DCF parameters and ASV errors), t_CM=CM threshold)')
        print('   tDCF_norm(t_CM) = {:8.5f} + {:8.5f} x Pmiss_cm(t_CM) + {:8.5f} x Pfa_cm(t_CM)\n'.format(
            C0 / tDCF_default, C1 / tDCF_default, C2 / tDCF_default))
        print(
            '     * The optimum value is given by the first term (0.06273). This is the normalized t-DCF obtained with an error-free CM system.')
        print('     * The minimum normalized cost (minimum over all possible thresholds) is always <= 1.00.')
        print('')

    return tDCF_norm, CM_thresholds


def compute_tDCF_legacy(bonafide_score_cm, spoof_score_cm, Pfa_asv, Pmiss_asv, Pmiss_spoof_asv, cost_model, print_cost):
    """
    Compute Tandem Detection Cost Function (t-DCF) [1] for a fixed ASV system.
    In brief, t-DCF returns a detection cost of a cascaded system of this form,

      Speech waveform -> [CM] -> [ASV] -> decision

    where CM stands for countermeasure and ASV for automatic speaker
    verification. The CM is therefore used as a 'gate' to decided whether or
    not the input speech sample should be passed onwards to the ASV system.
    Generally, both CM and ASV can do detection errors. Not all those errors
    are necessarily equally cost, and not all types of users are necessarily
    equally likely. The tandem t-DCF gives a principled with to compare
    different spoofing countermeasures under a detection cost function
    framework that takes that information into account.

    INPUTS:

      bonafide_score_cm   A vector of POSITIVE CLASS (bona fide or human)
                          detection scores obtained by executing a spoofing
                          countermeasure (CM) on some positive evaluation trials.
                          trial represents a bona fide case.
      spoof_score_cm      A vector of NEGATIVE CLASS (spoofing attack)
                          detection scores obtained by executing a spoofing
                          CM on some negative evaluation trials.
      Pfa_asv             False alarm (false acceptance) rate of the ASV
                          system that is evaluated in tandem with the CM.
                          Assumed to be in fractions, not percentages.
      Pmiss_asv           Miss (false rejection) rate of the ASV system that
                          is evaluated in tandem with the spoofing CM.
                          Assumed to be in fractions, not percentages.
      Pmiss_spoof_asv     Miss rate of spoof samples of the ASV system that
                          is evaluated in tandem with the spoofing CM. That
                          is, the fraction of spoof samples that were
                          rejected by the ASV system.
      cost_model          A struct that contains the parameters of t-DCF,
                          with the following fields.

                          Ptar        Prior probability of target speaker.
                          Pnon        Prior probability of nontarget speaker (zero-effort impostor)
                          Psoof       Prior probability of spoofing attack.
                          Cmiss_asv   Cost of ASV falsely rejecting target.
                          Cfa_asv     Cost of ASV falsely accepting nontarget.
                          Cmiss_cm    Cost of CM falsely rejecting target.
                          Cfa_cm      Cost of CM falsely accepting spoof.

      print_cost          Print a summary of the cost parameters and the
                          implied t-DCF cost function?

    OUTPUTS:

      tDCF_norm           Normalized t-DCF curve across the different CM
                          system operating points; see [2] for more details.
                          Normalized t-DCF > 1 indicates a useless
                          countermeasure (as the tandem system would do
                          better without it). min(tDCF_norm) will be the
                          minimum t-DCF used in ASVspoof 2019 [2].
      CM_thresholds       Vector of same size as tDCF_norm corresponding to
                          the CM threshold (operating point).

    NOTE:
    o     In relative terms, higher detection scores values are assumed to
          indicate stronger support for the bona fide hypothesis.
    o     You should provide real-valued soft scores, NOT hard decisions. The
          recommendation is that the scores are log-likelihood ratios (LLRs)
          from a bonafide-vs-spoof hypothesis based on some statistical model.
          This, however, is NOT required. The scores can have arbitrary range
          and scaling.
    o     Pfa_asv, Pmiss_asv, Pmiss_spoof_asv are in fractions, not percentages.

    References:

      [1] T. Kinnunen, K.-A. Lee, H. Delgado, N. Evans, M. Todisco,
          M. Sahidullah, J. Yamagishi, D.A. Reynolds: "t-DCF: a Detection
          Cost Function for the Tandem Assessment of Spoofing Countermeasures
          and Automatic Speaker Verification", Proc. Odyssey 2018: the
          Speaker and Language Recognition Workshop, pp. 312--319, Les Sables d'Olonne,
          France, June 2018 (https://www.isca-speech.org/archive/Odyssey_2018/pdfs/68.pdf)

      [2] ASVspoof 2019 challenge evaluation plan
          https://www.asvspoof.org/asvspoof2019/asvspoof2019_evaluation_plan.pdf
    """

    # Sanity check of cost parameters
    if cost_model['Cfa_asv'] < 0 or cost_model['Cmiss_asv'] < 0 or \
            cost_model['Cfa_cm'] < 0 or cost_model['Cmiss_cm'] < 0:
        print('WARNING: Usually the cost values should be positive!')

    if cost_model['Ptar'] < 0 or cost_model['Pnon'] < 0 or cost_model['Pspoof'] < 0 or \
            np.abs(cost_model['Ptar'] + cost_model['Pnon'] + cost_model['Pspoof'] - 1) > 1e-10:
        sys.exit('ERROR: Your prior probabilities should be positive and sum up to one.')

    # Unless we evaluate worst-case model, we need to have some spoof tests against asv
    if Pmiss_spoof_asv is None:
        sys.exit('ERROR: you should provide miss rate of spoof tests against your ASV system.')

    # Sanity check of scores
    combined_scores = np.concatenate((bonafide_score_cm, spoof_score_cm))
    if np.isnan(combined_scores).any() or np.isinf(combined_scores).any():
        sys.exit('ERROR: Your scores contain nan or inf.')

    # Sanity check that inputs are scores and not decisions
    n_uniq = np.unique(combined_scores).size
    if n_uniq < 3:
        sys.exit('ERROR: You should provide soft CM scores - not binary decisions')

    # Obtain miss and false alarm rates of CM
    Pmiss_cm, Pfa_cm, CM_thresholds = compute_det_curve(bonafide_score_cm, spoof_score_cm)

    # Constants - see ASVspoof 2019 evaluation plan
    C1 = cost_model['Ptar'] * (cost_model['Cmiss_cm'] - cost_model['Cmiss_asv'] * Pmiss_asv) - \
         cost_model['Pnon'] * cost_model['Cfa_asv'] * Pfa_asv
    C2 = cost_model['Cfa_cm'] * cost_model['Pspoof'] * (1 - Pmiss_spoof_asv)

    # Sanity check of the weights
    if C1 < 0 or C2 < 0:
        sys.exit(
            'You should never see this error but I cannot evalute tDCF with negative weights - please check whether your ASV error rates are correctly computed?')

    # Obtain t-DCF curve for all thresholds
    tDCF = C1 * Pmiss_cm + C2 * Pfa_cm

    # Normalized t-DCF
    tDCF_norm = tDCF / np.minimum(C1, C2)

    # Everything should be fine if reaching here.
    if print_cost:

        print(
            't-DCF evaluation from [Nbona={}, Nspoof={}] trials\n'.format(bonafide_score_cm.size, spoof_score_cm.size))
        print('t-DCF MODEL')
        print('   Ptar         = {:8.5f} (Prior probability of target user)'.format(cost_model['Ptar']))
        print('   Pnon         = {:8.5f} (Prior probability of nontarget user)'.format(cost_model['Pnon']))
        print('   Pspoof       = {:8.5f} (Prior probability of spoofing attack)'.format(cost_model['Pspoof']))
        print('   Cfa_asv      = {:8.5f} (Cost of ASV falsely accepting a nontarget)'.format(cost_model['Cfa_asv']))
        print(
            '   Cmiss_asv    = {:8.5f} (Cost of ASV falsely rejecting target speaker)'.format(cost_model['Cmiss_asv']))
        print(
            '   Cfa_cm       = {:8.5f} (Cost of CM falsely passing a spoof to ASV system)'.format(cost_model['Cfa_cm']))
        print('   Cmiss_cm     = {:8.5f} (Cost of CM falsely blocking target utterance which never reaches ASV)'.format(
            cost_model['Cmiss_cm']))
        print('\n   Implied normalized t-DCF function (depends on t-DCF parameters and ASV errors), s=CM threshold)')

        if C2 == np.minimum(C1, C2):
            print('   tDCF_norm(s) = {:8.5f} x Pmiss_cm(s) + Pfa_cm(s)\n'.format(C1 / C2))
        else:
            print('   tDCF_norm(s) = Pmiss_cm(s) + {:8.5f} x Pfa_cm(s)\n'.format(C2 / C1))

    return tDCF_norm, CM_thresholds


def load_asv_metrics_LA(asv_key_file, asv_scr_file, phase):
    # Load organizers' ASV scores
    asv_key_data = pandas.read_csv(asv_key_file, sep=' ', header=None)
    asv_scr_data = pandas.read_csv(asv_scr_file, sep=' ', header=None)[asv_key_data[7] == phase]
    idx_tar = asv_key_data[asv_key_data[7] == phase][5] == 'target'
    idx_non = asv_key_data[asv_key_data[7] == phase][5] == 'nontarget'
    idx_spoof = asv_key_data[asv_key_data[7] == phase][5] == 'spoof'

    # Extract target, nontarget, and spoof scores from the ASV scores
    tar_asv = asv_scr_data[2][idx_tar]
    non_asv = asv_scr_data[2][idx_non]
    spoof_asv = asv_scr_data[2][idx_spoof]
    eer_asv, asv_threshold = compute_eer(tar_asv, non_asv)
    [Pfa_asv, Pmiss_asv, Pmiss_spoof_asv, Pfa_spoof_asv] = obtain_asv_error_rates(tar_asv, non_asv, spoof_asv,
                                                                                  asv_threshold)

    return Pfa_asv, Pmiss_asv, Pmiss_spoof_asv, Pfa_spoof_asv


def performance_LA(cm_scores, Pfa_asv, Pmiss_asv, Pfa_spoof_asv, cost_model, invert=False):
    bona_cm = cm_scores[cm_scores[5] == 'bonafide']['1_x'].values
    spoof_cm = cm_scores[cm_scores[5] == 'spoof']['1_x'].values

    if invert == False:
        eer_cm = compute_eer(bona_cm, spoof_cm)[0]
    else:
        eer_cm = compute_eer(-bona_cm, -spoof_cm)[0]

    if invert == False:
        tDCF_curve, _ = compute_tDCF(bona_cm, spoof_cm, Pfa_asv, Pmiss_asv, Pfa_spoof_asv, cost_model, False)
    else:
        tDCF_curve, _ = compute_tDCF(-bona_cm, -spoof_cm, Pfa_asv, Pmiss_asv, Pfa_spoof_asv, cost_model, False)

    min_tDCF_index = np.argmin(tDCF_curve)
    min_tDCF = tDCF_curve[min_tDCF_index]

    return min_tDCF, eer_cm


def eval_to_score_file_LA(score_file, asv_key_file, asv_scr_file, cm_key_file, cost_model, phase):
    Pfa_asv, Pmiss_asv, Pmiss_spoof_asv, Pfa_spoof_asv = load_asv_metrics_LA(asv_key_file, asv_scr_file, phase)
    cm_data = pandas.read_csv(cm_key_file, sep=' ', header=None)
    submission_scores = pandas.read_csv(score_file, sep=' ', header=None, skipinitialspace=True)

    if len(submission_scores) != len(cm_data):
        print('CHECK: submission has %d of %d expected trials.' % (len(submission_scores), len(cm_data)))
        exit(1)

    # check here for progress vs eval set
    cm_scores = submission_scores.merge(cm_data[cm_data[7] == phase], left_on=0, right_on=1, how='inner')
    min_tDCF, eer_cm = performance_LA(cm_scores, Pfa_asv, Pmiss_asv, Pfa_spoof_asv, cost_model)

    bona_cm = cm_scores[cm_scores[5] == 'bonafide']['1_x'].values
    spoof_cm = cm_scores[cm_scores[5] == 'spoof']['1_x'].values

    # out_data = "min_tDCF: %.4f\n" % min_tDCF
    # out_data += "eer: %.2f\n" % (100 * eer_cm)
    # print(out_data, end="")

    out_data = "[phase:{}({}+{})\tmin_tDCF: {:.6f}      eer: {:.6f} %".format((phase + ']').ljust(15), len(bona_cm),
                                                                              len(spoof_cm), min_tDCF, 100 * eer_cm)
    logging.info(out_data)

    # just in case that the submitted file reverses the sign of positive and negative scores
    min_tDCF2, eer_cm2 = performance_LA(cm_scores, Pfa_asv, Pmiss_asv, Pfa_spoof_asv, cost_model, invert=True)

    if min_tDCF2 < min_tDCF:
        logging.info(
            'CHECK: we negated your scores and achieved a lower min t-DCF. Before: %.3f - Negated: %.3f - your class labels are swapped during training... this will result in poor challenge ranking' % (
                min_tDCF, min_tDCF2))

    if min_tDCF == min_tDCF2:
        logging.info(
            'WARNING: your classifier might not work correctly, we checked if negating your scores gives different min t-DCF - it does not. Are all values the same?')

    return min_tDCF


def compute_tDCF_EER_LA(submit_file, key_dir, phase='eval'):
    asv_key_file = os.path.join(key_dir, 'ASV/trial_metadata.txt')
    asv_scr_file = os.path.join(key_dir, 'ASV/ASVTorch_Kaldi/score.txt')
    cm_key_file = os.path.join(key_dir, 'CM/trial_metadata.txt')

    Pspoof = 0.05
    cost_model = {
        'Pspoof': Pspoof,  # Prior probability of a spoofing attack
        'Ptar': (1 - Pspoof) * 0.99,  # Prior probability of target speaker
        'Pnon': (1 - Pspoof) * 0.01,  # Prior probability of nontarget speaker
        'Cmiss': 1,  # Cost of tandem system falsely rejecting target speaker
        'Cfa': 10,  # Cost of tandem system falsely accepting nontarget speaker
        'Cfa_spoof': 10,  # Cost of tandem system falsely accepting spoof
    }

    # submission_scores = pandas.read_csv(submit_file, sep=' ', header=None, skipinitialspace=True)

    eval_to_score_file_LA(submit_file, asv_key_file, asv_scr_file, cm_key_file, cost_model, 'progress')
    eval_to_score_file_LA(submit_file, asv_key_file, asv_scr_file, cm_key_file, cost_model, 'eval')
    eval_to_score_file_LA(submit_file, asv_key_file, asv_scr_file, cm_key_file, cost_model, 'hidden_track')


def eval_to_score_file_DF(score_file, cm_key_file, phase):
    cm_data = pandas.read_csv(cm_key_file, sep=' ', header=None)
    submission_scores = pandas.read_csv(score_file, sep=' ', header=None, skipinitialspace=True)
    if len(submission_scores) != len(cm_data):
        logging.info('CHECK: submission has %d of %d expected trials.' % (len(submission_scores), len(cm_data)))
        exit(1)

    if len(submission_scores.columns) > 2:
        logging.info(
            'CHECK: submission has more columns (%d) than expected (2). Check for leading/ending blank spaces.' % len(
                submission_scores.columns))
        exit(1)

    cm_scores = submission_scores.merge(cm_data[cm_data[7] == phase], left_on=0, right_on=1,
                                        how='inner')  # check here for progress vs eval set
    bona_cm = cm_scores[cm_scores[5] == 'bonafide']['1_x'].values
    spoof_cm = cm_scores[cm_scores[5] == 'spoof']['1_x'].values
    eer_cm = compute_eer(bona_cm, spoof_cm)[0]
    out_data = "[phase:{}({}+{})\teer: {:8.5f} %".format((phase + ']').ljust(15), len(bona_cm), len(spoof_cm),
                                                         100 * eer_cm)
    logging.info(out_data)
    # print(out_data)
    return eer_cm


def compute_EER_DF(submit_file, truth_dir, phase='eval'):
    cm_key_file = os.path.join(truth_dir, 'CM/trial_metadata.txt')

    # Pspoof = 0.05
    # cost_model = {
    #     'Pspoof': Pspoof,  # Prior probability of a spoofing attack
    #     'Ptar': (1 - Pspoof) * 0.99,  # Prior probability of target speaker
    #     'Pnon': (1 - Pspoof) * 0.01,  # Prior probability of nontarget speaker
    #     'Cmiss': 1,  # Cost of tandem system falsely rejecting target speaker
    #     'Cfa': 10,  # Cost of tandem system falsely accepting nontarget speaker
    #     'Cfa_spoof': 10,  # Cost of tandem system falsely accepting spoof
    # }

    eval_to_score_file_DF(submit_file, cm_key_file, 'progress')
    eval_to_score_file_DF(submit_file, cm_key_file, 'eval')
    eval_to_score_file_DF(submit_file, cm_key_file, 'hidden_track')


def load_asv_metrics_PA(asv_key_file, asv_scr_file, phase):
    # Load organizers' ASV scores
    asv_key_data = pandas.read_csv(asv_key_file, sep=' ', header=None)
    asv_scr_data = pandas.read_csv(asv_scr_file, sep=' ', header=None)[asv_key_data[6] == phase]
    idx_tar = asv_key_data[asv_key_data[6] == phase][4] == 'target'
    idx_non = asv_key_data[asv_key_data[6] == phase][4] == 'nontarget'
    idx_spoof = asv_key_data[asv_key_data[6] == phase][4] == 'spoof'

    # Extract target, nontarget, and spoof scores from the ASV scores
    tar_asv = asv_scr_data[2][idx_tar]
    non_asv = asv_scr_data[2][idx_non]
    spoof_asv = asv_scr_data[2][idx_spoof]
    eer_asv, asv_threshold = compute_eer(tar_asv, non_asv)
    [Pfa_asv, Pmiss_asv, Pmiss_spoof_asv, Pfa_spoof_asv] = obtain_asv_error_rates(tar_asv, non_asv, spoof_asv,
                                                                                  asv_threshold)

    return Pfa_asv, Pmiss_asv, Pmiss_spoof_asv, Pfa_spoof_asv


def performance_PA(cm_scores, Pfa_asv, Pmiss_asv, Pfa_spoof_asv, cost_model, invert=False):
    bona_cm = cm_scores[cm_scores[4] == 'bonafide']['1_x'].values
    spoof_cm = cm_scores[cm_scores[4] == 'spoof']['1_x'].values

    if invert == False:
        eer_cm = compute_eer(bona_cm, spoof_cm)[0]
    else:
        eer_cm = compute_eer(-bona_cm, -spoof_cm)[0]

    if invert == False:
        tDCF_curve, _ = compute_tDCF(bona_cm, spoof_cm, Pfa_asv, Pmiss_asv, Pfa_spoof_asv, cost_model, False)
    else:
        tDCF_curve, _ = compute_tDCF(-bona_cm, -spoof_cm, Pfa_asv, Pmiss_asv, Pfa_spoof_asv, cost_model, False)

    min_tDCF_index = np.argmin(tDCF_curve)
    min_tDCF = tDCF_curve[min_tDCF_index]

    return min_tDCF, eer_cm


def eval_to_score_file_PA(score_file, asv_key_file, asv_scr_file, cm_key_file, cost_model, phase):
    Pfa_asv, Pmiss_asv, Pmiss_spoof_asv, Pfa_spoof_asv = load_asv_metrics_PA(asv_key_file, asv_scr_file, phase)
    cm_data = pandas.read_csv(cm_key_file, sep=' ', header=None)
    submission_scores = pandas.read_csv(score_file, sep=' ', header=None, skipinitialspace=True)

    if len(submission_scores) != len(cm_data):
        logging.info('CHECK: submission has %d of %d expected trials.' % (len(submission_scores), len(cm_data)))
        exit(1)

    # check here for progress vs eval set
    cm_scores = submission_scores.merge(cm_data[cm_data[6] == phase], left_on=0, right_on=1, how='inner')
    min_tDCF, eer_cm = performance_PA(cm_scores, Pfa_asv, Pmiss_asv, Pfa_spoof_asv, cost_model)

    # out_data = "min_tDCF: %.4f\n" % min_tDCF
    # out_data += "eer: %.2f\n" % (100 * eer_cm)
    # print(out_data, end="")

    bona_cm = cm_scores[cm_scores[4] == 'bonafide']['1_x'].values
    spoof_cm = cm_scores[cm_scores[4] == 'spoof']['1_x'].values

    out_data = "[phase:{}({}+{})\tmin_tDCF: {:8.5f}      eer: {:8.5f} %".format((phase + ']').ljust(15), len(bona_cm),
                                                                                len(spoof_cm), min_tDCF, 100 * eer_cm)
    logging.info(out_data)

    return min_tDCF


def compute_tDCF_EER_PA(submit_file, truth_dir, phase='eval'):
    asv_key_file = os.path.join(truth_dir, 'ASV/trial_metadata.txt')
    asv_scr_file = os.path.join(truth_dir, 'ASV/ASVTorch_Kaldi/score.txt')
    cm_key_file = os.path.join(truth_dir, 'CM/trial_metadata.txt')

    Pspoof = 0.05
    cost_model = {
        'Pspoof': Pspoof,  # Prior probability of a spoofing attack
        'Ptar': (1 - Pspoof) * 0.99,  # Prior probability of target speaker
        'Pnon': (1 - Pspoof) * 0.01,  # Prior probability of nontarget speaker
        'Cmiss': 1,  # Cost of tandem system falsely rejecting target speaker
        'Cfa': 10,  # Cost of tandem system falsely accepting nontarget speaker
        'Cfa_spoof': 10,  # Cost of tandem system falsely accepting spoof
    }

    eval_to_score_file_PA(submit_file, asv_key_file, asv_scr_file, cm_key_file, cost_model, 'progress')
    eval_to_score_file_PA(submit_file, asv_key_file, asv_scr_file, cm_key_file, cost_model, 'eval')
    eval_to_score_file_PA(submit_file, asv_key_file, asv_scr_file, cm_key_file, cost_model, 'hidden_track_1')
    eval_to_score_file_PA(submit_file, asv_key_file, asv_scr_file, cm_key_file, cost_model, 'hidden_track_2')


def compute_tDCF_EER21(submit_file, access_type, key_dir):
    # key_dir = '/home/lzc/lzc/ASVspoof2021/ASVspoof2021data/{}-keys-stage-1/keys'.format(access_type)

    if access_type == 'LA':
        return compute_tDCF_EER_LA(submit_file, key_dir)
    if access_type == 'PA':
        return compute_tDCF_EER_PA(submit_file, key_dir)
    if access_type == 'DF':
        return compute_EER_DF(submit_file, key_dir)


if __name__ == "__main__":
    logger_init_basic()

    # compute_tDCF_EER_LA(
    #     '/home/lzc/lzc/ASVspoof2021/ASVspoof2021exp/GMM_ResNet_MPath_LFCC/20211204_114854_GMM_ResNet_MPath_LFCC_LA/AS21_GMM_ResNet_MPath_LFCC_LA_20211204_114854_score.txt',
    #     '/home/lzc/lzc/ASVspoof2021/ASVspoof2021data/LA-keys-stage-1/keys')
    #
    # compute_EER_DF(
    #     '/home/lzc/lzc/ASVspoof2021/ASVspoof2021exp/GMM_ResNet_MPath_LFCC/20211204_114854_GMM_ResNet_MPath_LFCC_LA/AS21_GMM_ResNet_MPath_LFCC_DF_20211204_114854_score.txt',
    #     '/home/lzc/lzc/ASVspoof2021/ASVspoof2021data/DF-keys-stage-1/keys')
    #
    # compute_tDCF_EER_PA(
    #     '/home/lzc/lzc/ASVspoof2021/ASVspoof2021exp/GMM_ResNet_2Path_LFCC/20211111_012543_GMM_ResNet_2Path_LFCC_PA/AS21_GMM_ResNet_2Path_LFCC_PA_ufm_20211111_012543_score.txt',
    #     '/home/lzc/lzc/ASVspoof2021/ASVspoof2021data/PA-keys-stage-1/keys')

    key_dir = '/home/labuser/lzc/ASVspoof/ASVspoof2021data/LA-keys-stage-1/keys'
    compute_tDCF_EER21(
        '/home/labuser/lzc/ASVspoof2021exp/MSGMM_ResNet_LFCC21NN/20231106_094333_MSGMM_ResNet_LFCC21NN_LA/AS21_MSGMM_ResNet_LFCC21NN_LA_KEY_score.txt',
        'LA', key_dir)

    # compute_tDCF_EER21(
    #     '/home/lzc/lzc/ASVspoof2021/ASVspoof2021exp/GMM_ResNet_MPath_LFCC/20211204_114854_GMM_ResNet_MPath_LFCC_LA/AS21_GMM_ResNet_MPath_LFCC_DF_20211204_114854_score.txt',
    #     'DF')
    #
    # compute_tDCF_EER21(
    #     '/home/lzc/lzc/ASVspoof2021/ASVspoof2021exp/GMM_ResNet_2Path_LFCC/20211104_235542_GMM_ResNet_2Path_LFCC_PA/AS21_GMM_ResNet_2Path_LFCC_PA_20211104_235542_score.txt',
    #     'PA')

    # submit_file = sys.argv[1]
    # truth_dir = sys.argv[2]
    # phase = sys.argv[3]
    #
    # if not os.path.isfile(submit_file):
    #     print("%s doesn't exist" % (submit_file))
    #     exit(1)
    #
    # if not os.path.isdir(truth_dir):
    #     print("%s doesn't exist" % (truth_dir))
    #     exit(1)
    #
    # if phase != 'progress' and phase != 'eval' and phase != 'hidden_track':
    #     print("phase must be either progress, eval, or hidden_track")
    #     exit(1)
    #
    # _ = eval_to_score_file(submit_file, cm_key_file)
