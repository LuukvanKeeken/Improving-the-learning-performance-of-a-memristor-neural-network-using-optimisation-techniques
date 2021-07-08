import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = "true"
import argparse
import time
import warnings
with warnings.catch_warnings():
    warnings.simplefilter( "ignore" )

    import nengo_dl
    from nengo.learning_rules import PES
    from nengo.params import Default
    from nengo.processes import WhiteSignal
    from sklearn.metrics import mean_squared_error

    from memristor_nengo.extras import *
    from memristor_nengo.learning_rules import mPES

    setup()

    # Should not be useful for NengoDL>=3.3.0
    # tf.compat.v1.disable_eager_execution()
    # tf.compat.v1.disable_control_flow_v2()

    parser = argparse.ArgumentParser()
    parser.add_argument( "-f", "--function", default="x",
                        help="The function to learn.  Default is x" )
    parser.add_argument( "-i", "--inputs", default=[ "sine", "sine" ], nargs="*", choices=[ "sine", "white" ],
                        help="The input signals [learning, testing].  Default is sine" )
    parser.add_argument( "-t", "--timestep", default=0.001, type=int )
    parser.add_argument( "-S", "--simulation_time", default=30, type=int )
    parser.add_argument( "-N", "--neurons", nargs="*", default=[ 10 ], action="store", type=int,
                        help="The number of neurons used in the Ensembles [pre, post, error].  Default is 10" )
    parser.add_argument( "-D", "--dimensions", default=3, type=int,
                        help="The number of dimensions of the input signal" )
    parser.add_argument( "-n", "--noise", nargs="*", default=0.15, type=float,
                        help="The noise on the simulated memristors [R_0, R_1, c, R_init]  Default is 0.15" )
    parser.add_argument( "-g", "--gain", default=1e4, type=float )  # default chosen by parameter search experiments
    parser.add_argument( "-l", "--learning_rule", default="mPES", choices=[ "mPES", "PES" ] )
    parser.add_argument( "-P", "--parameters", default=Default, type=float,
                        help="The parametrs of simualted memristors.  For now only the exponent c" )
    parser.add_argument( "-b", "--backend", default="nengo_dl", choices=[ "nengo_dl", "nengo_core" ] )
    parser.add_argument( "-o", "--optimisations", default="run", choices=[ "run", "build", "memory" ] )
    parser.add_argument( "-s", "--seed", default=None, type=int )
    parser.add_argument( "--plot", default=0, choices=[ 0, 1, 2, 3 ], type=int,
                        help="0: No visual output, 1: Show plots, 2: Save plots, 3: Save data" )
    parser.add_argument( "--verbosity", default=2, choices=[ 0, 1, 2 ], type=int,
                        help="0: No textual output, 1: Only numbers, 2: Full output" )
    parser.add_argument( "-pd", "--plots_directory", default="../data/",
                        help="Directory where plots will be saved.  Default is ../data/" )
    parser.add_argument( "-d", "--device", default="/gpu:0",
                        help="/cpu:0 or /gpu:[x]" )
    parser.add_argument( "-lt", "--learn_time", default=3 / 4, type=float )
    parser.add_argument( '--probe', default=1, choices=[ 0, 1, 2 ], type=int,
                        help="0: probing disabled, 1: only probes to calculate statistics, 2: all probes active" )

    
    parser.add_argument("--optim_alg", default=1, type=int,
                        help="0: use no optimisation algorithm, 1: Simulated Annealing-like change of pulse size noise, 2: algorithm based on Stochastic Gradient Descent with Momentum, involving an exponentially decaying average of past pes_delta values. Default is none.")
    parser.add_argument("--SA_starting_noise", default=0.3, type=float, help="Set the initial noise level for the exponent, when using the SA-like algorithm. Default is 0.15.")                    
    parser.add_argument("--SA_ending_noise", default=0.2, type=float, help="Set the final noise level for the exponent, when using the SA-like algorithm. Default is 0.")
    parser.add_argument("--SA_schedule", default=0, type=int, help="0: linear SA schedule, gradually changes the noise between the starting and ending levels, 1: SA schedule that exponentially adjusts the noise from the starting to ending level. Default is lin.")
    parser.add_argument("--SA_exp_base", default=0.9998, type=float, help = "Base value that is used in the exponential SA schedule. Default is 0.9998.")
    parser.add_argument("--adapt_pulsing", default=1, type=int, help="Set whether to use adaptive pulsing. Default is 0 (false).")
    parser.add_argument("--pulse_levels", default=300, type=int, help="When using adaptive pulsing, set the maximum amount of pulses.")
    parser.add_argument("--Momentum_decay_factor", default=0.2, type=float, help="The factor by which each past past pes_delta value decays. Default is 0.2.")


    # TODO read parameters from conf file https://docs.python.org/3/library/configparser.html
    args = parser.parse_args()
   
   
    optim_alg = args.optim_alg
    SA_starting_noise = args.SA_starting_noise
    SA_ending_noise = args.SA_ending_noise
    SA_schedule = args.SA_schedule
    SA_exp_base = args.SA_exp_base
    adapt_pulsing = args.adapt_pulsing
    pulse_levels = args.pulse_levels
    Momentum_decay_factor = args.Momentum_decay_factor

    seed = args.seed
    tf.random.set_seed( seed )
    np.random.seed( seed )
    function_string = "lambda x: " + args.function
    function_to_learn = eval( function_string )
    if len( args.inputs ) not in (1, 2):
        parser.error( 'Either give no values for action, or two, not {}.'.format( len( args.inputs ) ) )
    if len( args.inputs ) == 1:
        if args.inputs[ 0 ] == "sine":
            input_function_train = input_function_test = Sines( period=4 )
        if args.inputs[ 0 ] == "white":
            input_function_train = input_function_test = WhiteSignal( period=60, high=5, seed=seed )
    if len( args.inputs ) == 2:
        if args.inputs[ 0 ] == "sine":
            input_function_train = Sines( period=4 )
        if args.inputs[ 0 ] == "white":
            input_function_train = WhiteSignal( period=60, high=5, seed=seed )
        if args.inputs[ 1 ] == "sine":
            input_function_test = Sines( period=4 )
        if args.inputs[ 1 ] == "white":
            input_function_test = WhiteSignal( period=60, high=5, seed=seed )
    timestep = args.timestep
    sim_time = args.simulation_time
    if len( args.neurons ) not in range( 1, 3 ):
        parser.error( 'Either give no values for action, or one, or three, not {}.'.format( len( args.neurons ) ) )
    if len( args.neurons ) == 1:
        pre_n_neurons = post_n_neurons = error_n_neurons = args.neurons[ 0 ]
    if len( args.neurons ) == 2:
        pre_n_neurons = error_n_neurons = args.neurons[ 0 ]
        post_n_neurons = args.neurons[ 1 ]
    if len( args.neurons ) == 3:
        pre_n_neurons = args.neurons[ 0 ]
        post_n_neurons = args.neurons[ 1 ]
        error_n_neurons = args.neurons[ 2 ]
    dimensions = args.dimensions
    noise_percent = args.noise
    gain = args.gain
    exponent = args.parameters
    learning_rule = args.learning_rule
    backend = args.backend
    optimisations = args.optimisations
    progress_bar = False
    printlv1 = printlv2 = lambda *a, **k: None
    if args.verbosity >= 1:
        printlv1 = print
    if args.verbosity >= 2:
        printlv2 = print
        progress_bar = True
    plots_directory = args.plots_directory
    device = args.device
    probe = args.probe
    generate_plots = show_plots = save_plots = save_data = False
    if args.plot >= 1:
        generate_plots = True
        show_plots = True
        probe = 2
    if args.plot >= 2:
        save_plots = True
    if args.plot >= 3:
        save_data = True

    # TODO give better names to folders or make hierarchy
    if save_plots or save_data:
        dir_name, dir_images, dir_data = make_timestamped_dir( root=plots_directory + learning_rule + "/" )

    learn_time = int( sim_time * args.learn_time )
    n_neurons = np.amax( [ pre_n_neurons, post_n_neurons ] )
    if optimisations == "build":
        optimize = False
        sample_every = timestep
        simulation_discretisation = 1
    elif optimisations == "run":
        optimize = True
        sample_every = timestep
        simulation_discretisation = 1
    elif optimisations == "memory":
        optimize = False
        sample_every = timestep * 100
        simulation_discretisation = n_neurons
    printlv2( f"Using {optimisations} optimisation" )

    model = nengo.Network( seed=seed )
    with model:
        nengo_dl.configure_settings( inference_only=True )
        # Create an input node
        input_node = nengo.Node(
                output=SwitchInputs( input_function_train,
                                    input_function_test,
                                    switch_time=learn_time ),
                size_out=dimensions
                )
        
        # Shut off learning by inhibiting the error population
        stop_learning = nengo.Node( output=lambda t: t >= learn_time )
        
        # Create the ensemble to represent the input, the learned output, and the error
        pre = nengo.Ensemble( pre_n_neurons, dimensions=dimensions, seed=seed )
        post = nengo.Ensemble( post_n_neurons, dimensions=dimensions, seed=seed )
        error = nengo.Ensemble( error_n_neurons, dimensions=dimensions, radius=2, seed=seed )
        
        # Connect pre and post with a communication channel
        # the matrix given to transform is the initial weights found in model.sig[conn]["weights"]
        # the initial transform has not influence on learning because it is overwritten by mPES
        # the only influence is on the very first timesteps, before the error becomes large enough
        conn = nengo.Connection(
                pre.neurons,
                post.neurons,
                transform=np.zeros( (post.n_neurons, pre.n_neurons) )
                )
        
        # Apply the learning rule to conn
        if learning_rule == "mPES":
            conn.learning_rule_type = mPES(
                    optim_alg = optim_alg,
                    SA_starting_noise = SA_starting_noise,
                    SA_ending_noise = SA_ending_noise,
                    SA_schedule = SA_schedule,
                    SA_exp_base = SA_exp_base,
                    adapt_pulsing = adapt_pulsing,
                    pulse_levels = pulse_levels,
                    Momentum_decay_factor = Momentum_decay_factor,
                    noisy=noise_percent,
                    gain=gain,
                    seed=seed,
                    exponent=exponent )
        if learning_rule == "PES":
            conn.learning_rule_type = PES()
        printlv2( "Simulating with", conn.learning_rule_type )
        
        # Provide an error signal to the learning rule
        nengo.Connection( error, conn.learning_rule )
        
        # Compute the error signal (error = actual - target)
        nengo.Connection( post, error )
        
        # Subtract the target (this would normally come from some external system)
        nengo.Connection( pre, error, function=function_to_learn, transform=-1 )
        
        # Connect the input node to ensemble pre
        nengo.Connection( input_node, pre )
        
        nengo.Connection(
                stop_learning,
                error.neurons,
                transform=-20 * np.ones( (error.n_neurons, 1) ) )
        
        # essential ones are used to calculate the statistics
        if probe > 0:
            pre_probe = nengo.Probe( pre, synapse=0.01, sample_every=sample_every )
            post_probe = nengo.Probe( post, synapse=0.01, sample_every=sample_every )
        if probe > 1:
            input_node_probe = nengo.Probe( input_node, sample_every=sample_every )
            error_probe = nengo.Probe( error, synapse=0.01, sample_every=sample_every )
            learn_probe = nengo.Probe( stop_learning, synapse=None, sample_every=sample_every )
            weight_probe = nengo.Probe( conn, "weights", synapse=None, sample_every=sample_every )
            post_spikes_probe = nengo.Probe( post.neurons, sample_every=sample_every )
            if isinstance( conn.learning_rule_type, mPES ):
                pos_memr_probe = nengo.Probe( conn.learning_rule, "pos_memristors", synapse=None,
                                            sample_every=sample_every )
                neg_memr_probe = nengo.Probe( conn.learning_rule, "neg_memristors", synapse=None,
                                            sample_every=sample_every )

    # Create the Simulator and run it
    printlv2( f"Backend is {backend}, running on ", end="" )
    if backend == "nengo_core":
        printlv2( "CPU" )
        cm = nengo.Simulator( model, seed=seed, dt=timestep, optimize=optimize, progress_bar=progress_bar )
    if backend == "nengo_dl":
        printlv2( device )
        cm = nengo_dl.Simulator( model, seed=seed, dt=timestep, progress_bar=progress_bar, device=device )
    start_time = time.time()
    with cm as sim:
        for i in range( simulation_discretisation ):
            printlv2( f"\nRunning discretised step {i + 1} of {simulation_discretisation}" )
            sim.run( sim_time / simulation_discretisation )
    printlv2( f"\nTotal time for simulation: {time.strftime( '%H:%M:%S', time.gmtime( time.time() - start_time ) )} s" )

    if probe > 0:
        # essential statistics
        y_true = sim.data[ pre_probe ][ int( (learn_time / timestep) / (sample_every / timestep) ):, ... ]
        y_pred = sim.data[ post_probe ][ int( (learn_time / timestep) / (sample_every / timestep) ):, ... ]
        # MSE after learning
        printlv2( "MSE after learning [f(pre) vs. post]:" )
        mse = mean_squared_error( function_to_learn( y_true ), y_pred, multioutput='raw_values' )
        printlv1( mse.tolist() )
        # Correlation coefficients after learning
        correlation_coefficients = correlations( function_to_learn( y_true ), y_pred )
        printlv2( "Pearson correlation after learning [f(pre) vs. post]:" )
        printlv1( correlation_coefficients[ 0 ] )
        printlv2( "Spearman correlation after learning [f(pre) vs. post]:" )
        printlv1( correlation_coefficients[ 1 ] )
        printlv2( "Kendall correlation after learning [f(pre) vs. post]:" )
        printlv1( correlation_coefficients[ 2 ] )
        printlv2( "MSE-to-rho after learning [f(pre) vs. post]:" )
        printlv1( mse_to_rho_ratio( mse, correlation_coefficients[ 1 ] ) )

    if probe > 1:
        # Average
        printlv2( "Weights average after learning:" )
        printlv1( np.average( sim.data[ weight_probe ][ -1, ... ] ) )
        
        # Sparsity
        printlv2( "Weights sparsity at t=0 and after learning:" )
        printlv1( gini( sim.data[ weight_probe ][ 0 ] ), end=" -> " )
        printlv1( gini( sim.data[ weight_probe ][ -1 ] ) )

    plots = { }
    if generate_plots and probe > 1:
        plotter = Plotter( sim.trange( sample_every=sample_every ), post_n_neurons, pre_n_neurons, dimensions,
                        learn_time,
                        sample_every,
                        plot_size=(13, 7),
                        dpi=300,
                        pre_alpha=0.3
                        )
        plots[ "results_smooth" ] = plotter.plot_results( sim.data[ input_node_probe ], sim.data[ pre_probe ],
                                                        sim.data[ post_probe ],
                                                        error=
                                                        sim.data[ post_probe ] -
                                                        function_to_learn( sim.data[ pre_probe ] ),
                                                        smooth=True )
        plots[ "results" ] = plotter.plot_results( sim.data[ input_node_probe ], sim.data[ pre_probe ],
                                                sim.data[ post_probe ],
                                                error=
                                                sim.data[ post_probe ] -
                                                function_to_learn( sim.data[ pre_probe ] ),
                                                smooth=False )
        plots[ "post_spikes" ] = plotter.plot_ensemble_spikes( "Post", sim.data[ post_spikes_probe ],
                                                            sim.data[ post_probe ] )
        plots[ "weights" ] = plotter.plot_weight_matrices_over_time( sim.data[ weight_probe ], sample_every=sample_every )
        
        plots[ "testing_smooth" ] = plotter.plot_testing( function_to_learn( sim.data[ pre_probe ] ),
                                                        sim.data[ post_probe ],
                                                        smooth=True )
        plots[ "testing" ] = plotter.plot_testing( function_to_learn( sim.data[ pre_probe ] ), sim.data[ post_probe ],
                                                smooth=False )
        if n_neurons <= 10 and learning_rule == "mPES":
            plots[ "weights_mpes" ] = plotter.plot_weights_over_time( sim.data[ pos_memr_probe ],
                                                                    sim.data[ neg_memr_probe ] )
            plots[ "memristors" ] = plotter.plot_values_over_time( sim.data[ pos_memr_probe ], sim.data[ neg_memr_probe ],
                                                                value="resistance" )

    if save_plots:
        assert generate_plots and probe > 1
        
        for fig in plots.values():
            fig.savefig( dir_images + str( i ) + ".pdf" )
            # fig.savefig( dir_images + str( i ) + ".png" )
        
        print( f"Saved plots in {dir_images}" )

    if save_data:
        save_weights( dir_data, sim.data[ weight_probe ] )
        print( f"Saved NumPy weights in {dir_data}" )
        
        save_results_to_csv( dir_data, sim.data[ input_node_probe ], sim.data[ pre_probe ], sim.data[ post_probe ],
                            sim.data[ post_probe ] - function_to_learn( sim.data[ pre_probe ] ) )
        save_memristors_to_csv( dir_data, sim.data[ pos_memr_probe ], sim.data[ neg_memr_probe ] )
        print( f"Saved data in {dir_data}" )

    #     TODO save output txt with metrics

    if show_plots:
        assert generate_plots and probe > 1
        
        for fig in plots.values():
            fig.show()
