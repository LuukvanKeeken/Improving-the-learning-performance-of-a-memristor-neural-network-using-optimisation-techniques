import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = "true"

import argparse
from subprocess import run

from memristor_nengo.extras import *

parser = argparse.ArgumentParser()
parser.add_argument( "-a", "--averaging", type=int, required=True )
parser.add_argument( "-i", "--inputs", nargs="*", choices=[ "sine", "white" ] )
parser.add_argument( "-f", "--function", default="x" )
parser.add_argument( "-N", "--neurons", type=int )
parser.add_argument( "-D", "--dimensions", type=int )
parser.add_argument( "-g", "--gain", type=float )
parser.add_argument( "-l", "--learning_rule", choices=[ "mPES", "PES" ] )
parser.add_argument( "--directory", default="../data/" )
parser.add_argument( "-lt", "--learn_time", default=3 / 4, type=float )
parser.add_argument( "-d", "--device", default="/gpu:0" )
parser.add_argument("--optim_alg", default=1, type=int)
parser.add_argument("--SA_schedule", default=0, type=int)
parser.add_argument("--SA_starting_noise", default=0.3, type=float)
parser.add_argument("--SA_ending_noise", default=0.2, type=float)
parser.add_argument("--SA_exp_base", default=0.9998, type=float)
parser.add_argument("--adapt_pulsing", default=0, type=int)
parser.add_argument("--pulse_levels", default=500, type=int)
parser.add_argument("--Momentum_decay_factor", default=0.2, type=float)
args = parser.parse_args()

learning_rule = args.learning_rule
gain = args.gain
function = args.function
inputs = args.inputs
neurons = args.neurons
dimensions = args.dimensions
num_averaging = args.averaging
directory = args.directory
learn_time = args.learn_time
device = args.device
optim_alg = args.optim_alg
SA_schedule = args.SA_schedule
SA_starting_noise = args.SA_starting_noise
SA_ending_noise = args.SA_ending_noise
SA_exp_base = args.SA_exp_base
adapt_pulsing = args.adapt_pulsing
pulse_levels = args.pulse_levels
Momentum_decay_factor = args.Momentum_decay_factor

dir_name, dir_images, dir_data = make_timestamped_dir(
        root=directory + "averaging/" + str( learning_rule ) + "/" + function + "_" + str( inputs ) + "_"
             + str( neurons ) + "_" + str( dimensions ) + "_" + str( gain ) + "/" )
print( "Reserved folder", dir_name )

print( "Evaluation for", learning_rule )
print( "Averaging runs", num_averaging )

res_mse = [ ]
res_pearson = [ ]
res_spearman = [ ]
res_kendall = [ ]
res_mse_to_rho = [ ]
counter = 0
for avg in range( num_averaging ):
    counter += 1
    print( f"[{counter}/{num_averaging}] Averaging #{avg + 1}" )
    # result = run(
    #         [ "python", "mPES.py", "--verbosity", str( 1 ), "-D", str( dimensions ), "-l", str( learning_rule ),
    #           "-N", str( neurons ), "-f", str( function ), "-lt", str( learn_time ), "-g", str( gain ),
    #           "-d", str( device ) ]
    #         + [ "-i" ] + inputs,
    #         capture_output=True,
    #         universal_newlines=True )

    result = run(
            [ "python", "mPES.py", "--verbosity", str( 1 ), "-D", str( dimensions ), "-l", str( learning_rule ),
              "-N", str( neurons ), "-f", str( function ), "-lt", str( learn_time ), "-g", str( gain ),
              "-d", str( device ), "--optim_alg", str(optim_alg), "--SA_schedule", str(SA_schedule),
              "--SA_starting_noise", str(SA_starting_noise), "--SA_ending_noise", str(SA_ending_noise),
              "--SA_exp_base", str(SA_exp_base), "--pulse_levels", str(pulse_levels), "--Momentum_decay_factor", str(Momentum_decay_factor) ]
            + [ "-i" ] + inputs,
            capture_output=True,
            universal_newlines=True )
   

    
    # save statistics
    try:
        print('NO ERROR')
        mse = np.mean( [ float( i ) for i in result.stdout.split( "\n" )[ 0 ][ 1:-1 ].split( "," ) ] )
        print( "MSE", mse )
        res_mse.append( mse )
        pearson = np.mean( [ float( i ) for i in result.stdout.split( "\n" )[ 1 ][ 1:-1 ].split( "," ) ] )
        print( "Pearson", pearson )
        res_pearson.append( pearson )
        spearman = np.mean( [ float( i ) for i in result.stdout.split( "\n" )[ 2 ][ 1:-1 ].split( "," ) ] )
        print( "Spearman", spearman )
        res_spearman.append( spearman )
        kendall = np.mean( [ float( i ) for i in result.stdout.split( "\n" )[ 3 ][ 1:-1 ].split( "," ) ] )
        print( "Kendall", kendall )
        res_kendall.append( kendall )
        mse_to_rho = np.mean( [ float( i ) for i in result.stdout.split( "\n" )[ 4 ][ 1:-1 ].split( "," ) ] )
        print( "MSE-to-rho", mse_to_rho )
        res_mse_to_rho.append( mse_to_rho )
    except:
        print('ERROR')
        if ('W tensorflow/core/common_runtime/gpu/gpu_bfc_allocator.cc:39] Overriding allow_growth setting because the TF_FORCE_GPU_ALLOW_GROWTH environment variable is set. Original config value was 0.' in result.stderr):
            print(result.stdout)
            print(len(result.stdout.split("\n")))
            mse = np.mean( [ float( i ) for i in result.stdout.split( "\n" )[ 0 ][ 1:-1 ].split( "," ) ] )
            print( "MSE", mse )
            res_mse.append( mse )
            pearson = np.mean( [ float( i ) for i in result.stdout.split( "\n" )[ 1 ][ 1:-1 ].split( "," ) ] )
            print( "Pearson", pearson )
            res_pearson.append( pearson )
            spearman = np.mean( [ float( i ) for i in result.stdout.split( "\n" )[ 2 ][ 1:-1 ].split( "," ) ] )
            print( "Spearman", spearman )
            res_spearman.append( spearman )
            kendall = np.mean( [ float( i ) for i in result.stdout.split( "\n" )[ 3 ][ 1:-1 ].split( "," ) ] )
            print( "Kendall", kendall )
            res_kendall.append( kendall )
            mse_to_rho = np.mean( [ float( i ) for i in result.stdout.split( "\n" )[ 4 ][ 1:-1 ].split( "," ) ] )
            print( "MSE-to-rho", mse_to_rho )
            res_mse_to_rho.append( mse_to_rho )
        else:
            print( "Ret", result.returncode )
            print( "Out", result.stdout )
            print( "Err", result.stderr )


mse_means = np.mean( res_mse )
pearson_means = np.mean( res_pearson )
spearman_means = np.mean( res_spearman )
kendall_means = np.mean( res_kendall )
mse_to_rho_means = np.mean( res_mse_to_rho )
print( "Average MSE:", mse_means )
print( "Average Pearson:", pearson_means )
print( "Average Spearman:", spearman_means )
print( "Average Kendall:", kendall_means )
print( "Average MSE-to-rho:", mse_to_rho_means )

res_list = range( num_averaging )

fig = plt.figure()
ax = fig.add_subplot( 111 )
ax.plot( res_list, res_mse, label="MSE" )
ax.legend()
fig.savefig( dir_images + "mse" + ".pdf" )

fig = plt.figure()
ax = fig.add_subplot( 111 )
ax.plot( res_list, res_pearson, label="Pearson" )
ax.plot( res_list, res_spearman, label="Spearman" )
ax.plot( res_list, res_kendall, label="Kendall" )
ax.legend()
fig.savefig( dir_images + "correlations" + ".pdf" )

print( f"Saved plots in {dir_images}" )

np.savetxt( dir_data + "results.csv",
            np.stack( (res_mse, res_pearson, res_spearman, res_kendall), axis=1 ),
            delimiter=",", header="MSE,Pearson,Spearman,Kendall", comments="" )
with open( dir_data + "parameters.txt", "w" ) as f:
    f.write( f"Learning rule: {learning_rule}\n" )
    f.write( f"Function: {function}\n" )
    f.write( f"Neurons: {neurons}\n" )
    f.write( f"Dimensions: {dimensions}\n" )
    f.write( f"Number of runs for averaging: {num_averaging}\n" )
    f.write( f"Optimisation algorithm: {optim_alg}\n")
    f.write( f"Schedule: {SA_schedule}\n")
    f.write( f"Starting noise: {SA_starting_noise}\n")
    f.write( f"Ending noise: {SA_ending_noise}\n")
    f.write( f"Exp_base: {SA_exp_base}\n")
    f.write( f"Pulse levels: {pulse_levels}\n")
    f.write( f"Momentum decay factor: {Momentum_decay_factor}")


with open( dir_data + "output.csv", "w" ) as f2:
    f2.write( f"Average MSE: {mse_means}\n")
    f2.write( f"Average Pearson: {pearson_means}\n")
    f2.write( f"Average Spearman: {spearman_means}\n")
    f2.write( f"Average Kendall: {kendall_means}\n")
    f2.write( f"Average MSE-to-rho: {mse_to_rho_means}\n")
    f2.write( f"Average rho/Average MSE: {spearman_means/mse_means}\n")
print( f"Saved data in {dir_data}" )