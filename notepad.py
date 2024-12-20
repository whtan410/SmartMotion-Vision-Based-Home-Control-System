import argparse

if __name__ == "__main__" :
	parser = argparse.ArgumentParser()
	parser.add_argument("--phy", type=int , help='Physic score')
	parser.add_argument("--chem", type=int , help='Chemistry score')
	parser.add_argument("--math", type=int , help-'Maths score')
	args = parser.parse_args()
	mean = (args.phy + args.chem + args.math)/3  
	
	print("Average: ", mean)