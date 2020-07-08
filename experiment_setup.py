"""
Hyper-parameters
"""
import argparse


def get_arguments():
    """ Hyper-parameters """

    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--architecture", type=str, choices=['reim',
                                                                   'reim2x',
                                                                   'reimsqrt2x',
                                                                   'magnitude',
                                                                   'phase',
                                                                   're',
                                                                   'im',
                                                                   'modrelu',
                                                                   'crelu'],
                        default='modrelu',
                        help="Architecture")

    setup = parser.add_argument_group("setup", "Setup for experiments")

    setup.add_argument("-phy_ch", "--physical_channel", action="store_true",
                       help="Train network, default = False"
                       )
    setup.add_argument("-phy_cfo", "--physical_cfo", action="store_true",
                       help="Attack network, default = False",
                       )
    setup.add_argument("-comp_cfo", "--compensate_cfo", action="store_true",
                       help="Attack network, default = False",
                       )
    setup.add_argument("-eq_tr", "--equalize_train", action="store_true", default=False,
                       help="For Saving the current Model, default = False ",
                       )

    setup.add_argument("-aug_ch", "--augment_channel", action="store_true", default=False,
                       help="For Saving the current Model, default = False ",
                       )
    setup.add_argument("-aug_cfo", "--augment_cfo", action="store_true", default=False,
                       help="For Saving the current Model, default = False ",
                       )
    setup.add_argument("-res", "--obtain_residuals", action="store_true", default=False,
                       help="For Saving the current Model, default = False ",
                       )

    test_setup = parser.add_argument_group("test setup", "Test Setup for experiments")

    test_setup.add_argument("-eq_test", "--equalize_test", action="store_true", default=False,
                            help="For Saving the current Model, default = False ",
                            )
    setup.add_argument("-comp_cfo_test", "--compensate_cfo_test", action="store_true",
                       help="Attack network, default = False",
                       )

    test_setup.add_argument("-aug_ch_test", "--augment_channel_test", action="store_true", default=False,
                            help="For Saving the current Model, default = False ",
                            )
    test_setup.add_argument("-aug_cfo_test", "--augment_cfo_test", action="store_true", default=False,
                            help="For Saving the current Model, default = False ",
                            )

    args = parser.parse_args()

    return args
