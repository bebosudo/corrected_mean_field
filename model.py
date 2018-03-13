import sympy as sp
import numpy as np
from sympy.printing.theanocode import theano_function
# from sympy.tensor.array import DenseNDimArray


####################################################################################

class ModelError(ValueError):
    def __init__(self, expr, base_exc=None):
        self.expr = expr
        self.base_exc = base_exc

    def __str__(self):
        if self.base_exc is None:
            return "ModelError: %r" % (self.expr,)

        return ("Model generation failed:\n%s: %s" % (self.expr, self.base_exc.__class__.__name__,
            str(self.base_exc)))


####################################################################################


class Transition:

    def __init__(self, update, rate, sympy_symbols):
        #this will convert rate into a sympy expression.
        #we need a sanity check before?
        self.update = None
        self.update_dictionary = update
        try:
            self.rate = sp.sympify(rate, sympy_symbols, evaluate=False)
        except sp.SympifyError as e:
            print("An error happened while parsing expression", rate,":",e)

    # finalizes transition by turning the update list into a numpy array
    def finalize(self, variables):
        self.update = np.zeros(variables.dimension)
        for var_name in self.update_dictionary:
            index = variables.get_id(var_name)
            self.update[index] = self.update_dictionary[var_name]
        self.update = np.reshape(self.update, (1,variables.dimension))
        #print("transition; update is", self.update)

####################################################################################


class Observable:

    def __init__(self, observable_function, sympy_symbols):
        try:
            self.function = sp.sympify(observable_function, sympy_symbols, evaluate=False)
        except sp.SympifyError as e:
            print("An error happened while parsing expression", observable_function, ":", e)

    # finalizes transition by turning the update list into a numpy array
    def finalize(self, variables):
        n = variables.dimension
        x = variables.reference
        self.gradient = np.zeros((1, n), dtype=object)
        self.hessian = np.zeros((n, n), dtype=object)
        for i in range(n):
            self.gradient[0,i] = sp.diff(self.function, x[i])
            for j in range(i,n):
                self.hessian[i, j] = sp.diff(self.function, x[i], x[j])
                self.hessian[j, i] = self.hessian[i, j]





####################################################################################

class Symbols:

    def __init__(self):
        # initialize
        self.reference = []
        self.values = []
        # initialize map from string to int and back
        self.reference2id = {}
        self.names2id = {}
        self.id2reference = {}
        self.dimension = 0

    #Adds a new symbol to the symbol array, creating a sympy object
    def add(self, name, value):
        symbol = sp.symbols(name)
        index = len(self.reference)
        self.reference.append(symbol)
        self.values.append(value)
        self.reference2id[symbol] = index
        self.names2id[name] = index
        self.id2reference[index] = symbol
        self.dimension += 1
        return symbol

    # sets the value of a symbol
    def set(self, name, value):
        try:
            index = self.names2id[name]
            self.values[index] = value
        except:
            print("Symbol " + name + " is not defined")

    def get_value(self,name):
        try:
            index = self.names2id[name]
            return self.values[index]
        except:
            print("Symbol " + name + " is not defined")

    #finalizes the symbol array, generating a numpy array for values
    def finalize(self):
        self.values = np.array(self.values)

    def get_id(self, name):
        return self.names2id[name]


####################################################################################


class Model:

    def __init__(self):
        #init variables and parameters
        self.variables = Symbols()
        self.parameters = Symbols()
        # this contains a map of names to sympy variables, to be used later for parsing expressions
        self.names2sym = {}
        # init transition list
        self.transitions = []
        self.transition_number = 0
        self.observable_list = []
        self.observable_dimension = 0
        self.system_size = 0;
        self.system_size_reference = None
        self.system_size_name = ''
        self.observable_names = []

    def set_system_size(self, name, value):
        self.add_parameter(name, value)
        self.system_size_reference = self.names2sym[name]
        self.system_size_name = name
        self.system_size = value


    def add_variable(self, name, value):
        if name in self.names2sym:
            raise ModelError("Name " + name + " already defined!")
        var = self.variables.add(name, value)
        self.names2sym[name] = var

    def add_parameter(self, name, value):
        if name in self.names2sym:
            raise ModelError("Name " + name + " already defined!")
        par = self.parameters.add(name, value)
        self.names2sym[name] = par

    # Changes the initial value of a variable
    def set_variable(self, name, value):
        self.variables.set(name, value)

    # Changes the value of a parameter
    def set_parameter(self, name, value):
        if self.system_size_name == name:
            self.parameters.set(name, value)
            self.system_size = value
        else:
            self.parameters.set(name, value)

    def get_parameter_value(self, name):
        return self.parameters.get_value(name)

    # Adds a transition to the model
    def add_transition(self, update, rate):
        t = Transition(update, rate, self.names2sym)
        self.transitions.append(t)
        self.transition_number += 1

    def add_observable(self, obs_name, observable):
        obs = Observable(observable, self.names2sym)
        self.observable_list.append(obs)
        self.observable_names.append(obs_name)
        self.observable_dimension += 1

    # Finalizes the initialization
    def finalize_initialization(self):
        self.variables.finalize()
        self.parameters.finalize()
        for t in self.transitions:
            t.finalize(self.variables)
        for obs in self.observable_list:
            obs.finalize(self.variables)
        self.__generate_vector_field()
        self.__generate_diffusion()
        self.__generate_jacobian()
        self.__generate_hessian()
        self.__generate_observable_functions()
        self.__generate_numpy_functions()

    #generates the mean field vector field
    def __generate_vector_field(self):
        self._vector_field_sympy = np.zeros(self.transitions[0].update.shape, dtype=object)
        for t in self.transitions:
            self._vector_field_sympy += t.update * t.rate
        self._vector_field_sympy = sp.simplify(self._vector_field_sympy)
        # for i in range(self._vector_field_sympy.size):
        #     v = self._vector_field_sympy[0, i]
        #     self._vector_field_sympy[0, i] = sp.simplify(v)
        # print("Vector field")
        # print(self._vector_field_sympy)

    #generates the diffusion term
    def __generate_diffusion(self):
        self._diffusion_sympy = 0
        for t in self.transitions:
            self._diffusion_sympy += np.matmul(np.transpose(t.update), t.update) * t.rate
        self._diffusion_sympy = sp.simplify(self._diffusion_sympy)
        # for i in range(self._vector_field_sympy.shape[0]):
        #     for j in range(self._vector_field_sympy.shape[1]):
        #         self._diffusion_sympy[i,j] = sp.simplify(self._diffusion_sympy[i,j])
        # print("Diffusion")
        # print(self._diffusion_sympy)

    # computes symbolically the jacobian of the vector field
    def __generate_jacobian(self):
        n = self.variables.dimension
        f = self._vector_field_sympy
        x = self.variables.reference
        J = np.zeros((n, n), dtype=object)
        for i in range(n):
            for j in range(n):
                J[i, j] = sp.diff(f[0, i], x[j])
        self._jacobian_sympy = J
        #print("Jacobian")
        #print(J)

    #computes the Hessian of the vector field
    def __generate_hessian(self):
        n = self.variables.dimension
        f = self._vector_field_sympy
        x = self.variables.reference
        H = np.zeros((n, n, n), dtype=object)
        for k in range(n): #function
            for i in range(n): #first variable to differentiate
                for j in range(i,n): #second variable to differentiate
                    H[k, i, j] = sp.diff(f[0,k],x[i],x[j])
                    H[k, j, i] = H[k, i, j] #by symmetry of derivatives
        self._hessian_sympy = H
        #print("Hessian")
        #print(H)


    def __generate_observable_functions(self):
        n = self.variables.dimension
        p = len(self.observable_list)
        self._observables_sympy = np.zeros((1, p), dtype=object)
        self._observables_jacobian_sympy = np.zeros((p,n), dtype=object)
        self._observables_hessian_sympy = np.zeros((p, n, n), dtype=object)
        for k in range(p):
            obs = self.observable_list[k]
            self._observables_sympy[0,k] = obs.function
            self._observables_jacobian_sympy[k,:] = obs.gradient
            self._observables_hessian_sympy[k,:,:] =obs.hessian
        #print("observables")
        #print(self._observables_sympy)
        #print("obs jacobian")
        #print(self._observables_jacobian_sympy)
        #print("obs hessian")
        #print(self._observables_hessian_sympy)


    # generate numpy expressions and the mean field VF
    def __generate_numpy_functions(self):
        sympy_ref = self.variables.reference + self.parameters.reference
        self.rates = sp.lambdify(sympy_ref, [t.rate for t in self.transitions], "numpy")
        self.vector_field = sp.lambdify(sympy_ref, self._vector_field_sympy, "numpy")
        self.diffusion = sp.lambdify(sympy_ref, self._diffusion_sympy, "numpy")
        self.jacobian = sp.lambdify(sympy_ref, self._jacobian_sympy, "numpy")
        self.hessian = sp.lambdify(sympy_ref, self._hessian_sympy, "numpy")
        self.observables = sp.lambdify(sympy_ref, self._observables_sympy, "numpy")
        self.observable_jacobian = sp.lambdify(sympy_ref, self._observables_jacobian_sympy, "numpy")
        self.observable_hessian = sp.lambdify(sympy_ref, self._observables_hessian_sympy, "numpy")


    #evaluates and returns vector field, diffusion, ...
    def evaluate_all_vector_fields(self, var_values):
        f = self.vector_field(*var_values, *self.parameters.values)
        G = self.diffusion(*var_values, *self.parameters.values)
        J = self.jacobian(*var_values, *self.parameters.values)
        H = self.hessian(*var_values, *self.parameters.values)
        return np.asarray(f), np.asarray(G), np.asarray(J), np.asarray(H)

    def evaluate_MF_vector_field(self, var_values):
        f = self.vector_field(*var_values, *self.parameters.values)
        return np.asarray(f)

    # evaluates and returns observables and their jacobian and hessian ...
    def evaluate_all_observables(self, var_values):
        g = self.observables(*var_values, *self.parameters.values)
        J = self.observable_jacobian(*var_values, *self.parameters.values)
        H = self.observable_hessian(*var_values, *self.parameters.values)
        return np.asarray(g), np.asarray(J), np.asarray(H)

    # evaluates and returns observables and their jacobian and hessian ...
    def evaluate_observables(self, var_values):
        g = self.observables(*var_values, *self.parameters.values)
        return np.asarray(g)

    def evaluate_rates(self, var_values):
        r = self.rates(*var_values, *self.parameters.values)
        return np.asarray(r)

    # Generates CERENA files from specification
    def generate_CERENA(self, model_name, directory, time):

        # Model definition file
        modelDefFile = model_name + "_def"
        with open(directory + "/" + modelDefFile + ".m", 'w') as f:

            f.write("% System variables\nsyms ");
            f.write(" ".join(str(variable) for variable in self.variables.reference))
            f.write("\n% System parameters\nsyms ");
            f.write(" ".join(str(param) for param in self.parameters.reference))
            f.write("\nsyms Omega time\n")
            f.write("System.time = time;\n")
            f.write("System.compartments = {'cell'};\n")
            f.write("System.volumes = Omega;\n")
            # Declaration of system variables
            f.write("System.state.variable = [");
            f.write("; ".join(str(variable) for variable in self.variables.reference))
            f.write("];\n")

            # Declaration of compartments -- this is always the same
            f.write("System.state.compartment = {");
            f.write("'cell';" * self.variables.dimension)
            f.write("};\n")

            # Initial conditions
            f.write("System.state.mu0 = [");
            f.write("; ".join(str(value) for value in self.variables.values))
            f.write("];\n")

            # Parameters
            f.write("% Parameters\n")
            f.write("System.parameter.variable = [");
            f.write("; ".join(str(param) for param in self.parameters.reference))
            f.write("];\n")
            f.write("System.kappa.variable = Omega;\n")
            f.write("System.scaleIndicator = 'microscopic';\n")

            # Output variables
            # Only works if the observal function is a variable
            f.write("System.output.variable = [")
            f.write("; ".join(str(obs.function) for obs in self.observable_list))
            f.write("];\n")
            f.write("System.output.function = [")
            f.write("; ".join(str(obs.function) for obs in self.observable_list))
            f.write("];\n")

            i = 1;
            for transition in self.transitions:
                educts = []
                products = []
                for j in range(len(transition.update[0])):
                    if transition.update[0][j] < 0:
                        educts.append(self.variables.reference[j])
                    if transition.update[0][j] > 0:
                        products.append(self.variables.reference[j])
                f.write("System.reaction({}).educt = {};\n".format(i,educts))
                f.write("System.reaction({}).product = {};\n".format(i, products))
                f.write("System.reaction({}).propensity = {};\n".format(i, transition.rate))
                i = i + 1

            f.write("t = linspace(0,{},100);\n".format(time))
            # Printing the values of the parameters
            f.write("theta = [")
            f.write("; ".join(str(value) for value in self.parameters.values))
            f.write("];\n")
            # Printing the values of the Omega parameter, fixed to one
            f.write("kappa = 1;\n")

        # Simulation file
        with open(directory + "/" + model_name + "_sim.m", 'w') as f:
            f.write("modelDefName = '" + model_name + "_def';\n")
            f.write("modelName = '{}_EMRE';\n".format(model_name))
            f.write("method = 'EMRE';\n")
            f.write("System_EMRE = genSimFile(modelName, modelDefName, method);\n")
            f.write("System_EMRE.sol = simulate_{}_EMRE(t,theta,kappa);\n".format(model_name))
            f.write("output = [System_EMRE.sol.t' System_EMRE.sol.y(:,1:{size})];\n".format(size=len(self.observable_list)))
            f.write("csvwrite('{}_emre.csv', output)\n".format(model_name));


    ### this code is working badly: Theano is very slow, probably badly configured in my machine


    def __generate_theano_functions(self):
        self.rates_theano = theano_function(self.variables.reference + self.parameters.reference,
                                            [t.rate for t in self.transitions], on_unused_input='ignore')
        self.vector_field_theano = theano_function(self.variables.reference + self.parameters.reference,
                                                             self._vector_field_sympy, on_unused_input='ignore')

    def evaluate_rates_theano(self, vars):
        r = self.rates_theano(*vars, *self.parameters.values)
        return r

