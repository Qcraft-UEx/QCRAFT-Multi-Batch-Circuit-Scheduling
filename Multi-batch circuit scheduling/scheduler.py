from urllib.parse import urlparse
import json
import ast
from urllib.parse import unquote
from flask import Flask, request
import socket
import requests
from divideResults import divideResults
import logging
import uuid
import re
from scheduler_policies import SchedulerPolicies
from executeCircuitIBM import executeCircuitIBM
from executeCircuitAWS import retrieve_result_aws, code_to_circuit_aws
from executeCircuitMicrosoft import executeCircuitMicrosoft  # <-- nuevo para Microsoft/Qiskit
import os
from threading import Thread, Lock
from flask import jsonify
from pymongo import MongoClient
from bson.json_util import dumps
from dotenv import load_dotenv
from qiskit import Aer, execute, transpile
from qiskit.circuit import QuantumCircuit

class Scheduler:
    """
    Class to manage the petitions of quantum circuit scheduling.
    """
    def __init__(self):
        """
        Initialize the scheduler
        """
        self.app = Flask(__name__)
        self.ports = {}

        dotenv_path = os.path.join(os.path.dirname(__file__), 'db', '.env')
        load_dotenv(dotenv_path)

        self.app.config['HOST'] = os.getenv('HOST')
        self.app.config['PORT'] = os.getenv('PORT')
        self.app.config['TRANSLATOR'] = os.getenv('TRANSLATOR')
        self.app.config['TRANSLATOR_PORT'] = os.getenv('TRANSLATOR_PORT')
        self.app.config['DB'] = os.getenv('DB')
        self.app.config['DB_PORT'] = os.getenv('DB_PORT')

        #mongo_uri = f"mongodb://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}@{self.app.config['DB']}:{self.app.config['DB_PORT']}/"
        #self.client = MongoClient(mongo_uri)
        #self.db = self.client[os.getenv('DB_NAME')]
        #self.collection = self.db[os.getenv('DB_COLLECTION')]

        self.translator = f"http://{self.app.config['TRANSLATOR']}:{self.app.config['TRANSLATOR_PORT']}/code/"
        self.policy_service = f"http://{self.app.config['HOST']}:{self.app.config['PORT']}/service/"

        self.scheduler_policies = SchedulerPolicies(self.app)
        #self.max_qubits = 127

        self.executeCircuitIBM = self.scheduler_policies.get_ibm()
        self.executeCircuitMicrosoft = executeCircuitMicrosoft()  # <-- inicialización Microsoft

        self.transpilation_machine = self.scheduler_policies.get_ibm_machine()
        self.service = self.executeCircuitIBM.load_account_ibm()
        if self.transpilation_machine != 'local':
            self.transpilation_backend = self.executeCircuitIBM.obtain_machine(self.service, self.transpilation_machine)

        self.app.route('/url', methods=['POST'])(self.store_url)
        self.app.route('/circuit', methods=['POST'])(self.store_url_circuit)
        self.app.route('/unscheduler', methods=['POST'])(self.unschedule_route)
        self.app.route('/result', methods=['GET'])(self.sendResults)

        self.result_lock = Lock()
        Thread(target=self.check_ids).start()

        @self.app.errorhandler(404)
        def not_found_error(error):
            return 'This route does not exist', 404

        @self.app.errorhandler(500)
        def internal_error(error):
            return 'Internal server error', 500

    def run(self) -> None:
        self.updatePorts()
        print('hecho')
        self.app.run(host='0.0.0.0', port=self.app.config['PORT'], debug=False)

    def handle_line(self, line:str, ids_file:str, lock:Lock) -> None:
        fdata = json.loads(line)
        id = list(fdata.keys())[0]
        users = fdata[id][0]
        qubit_number = fdata[id][1]
        shots = fdata[id][2]
        provider = fdata[id][3]
        if provider == 'ibm':
            counts = self.executeCircuitIBM.retrieve_result_ibm(id)
        elif provider == 'aws':
            counts = retrieve_result_aws(id)
        elif provider == 'microsoft':
            # Aquí podrías implementar la recuperación si fuera necesario
            counts = {}
        circuit_names = fdata[id][4]
        self.unscheduler(counts, shots, provider, qubit_number, users, circuit_names)
        with lock:
            with open(ids_file, 'r') as file:
                lines = file.readlines()
            with open(ids_file, 'w') as file:
                for line in lines:
                    line_dict = json.loads(line.strip())
                    if list(line_dict.keys())[0] != id:
                        file.write(line)

    def check_ids(self) -> None:
        script_dir = os.path.dirname(os.path.realpath(__file__))
        ids_file = os.path.join(script_dir, 'ids.txt')
        with open(ids_file, 'r') as file:
            lines = file.readlines()
        
        lock = Lock()
        threads = []
        for line in lines:
            t = Thread(target=self.handle_line, args=(line, ids_file, lock))
            t.start()
            threads.append(t)

    def select_policy(self, url:str, num_qubits:int, shots:int, user:int, circuit_name:str, maxDepth:int, provider:str, policy:str, criterio:str) -> None:
        data = {"circuit": url, "num_qubits": num_qubits, "shots": shots, "user": user, "circuit_name": circuit_name, "maxDepth": maxDepth, "provider": provider, "criterio": criterio}
        requests.post(self.policy_service+policy, json=data)

    def unschedule_route(self) -> tuple:
        data = request.get_json()
        self.unscheduler(data['counts'], data['shots'], data['provider'], data['qb'], data['users'], data['circuit_names'])
        return jsonify({'status': 'success'}), 200

    def unscheduler(self, counts:dict, shots:int, provider:str, qb:list, users:list, circuit_names:list) -> tuple:
        results = divideResults(counts, shots, provider, qb, users, circuit_names)
        for dividedResult in results:
            for key, value in dividedResult.items():
                id, circuit_name = key
                update = {'$inc': {'value.' + k: v for k, v in value.items()}}
                # with self.result_lock:
                #     self.collection.update_one({'_id': str(id), 'circuit': circuit_name}, update, upsert=True)
        return "Results stored successfully", 200

    def store_url(self) -> tuple:
        if request.json.get('url') is None:
            return "URL must be specified", 400
        provider = request.json.get('provider', 'ibm')
        policy = request.json.get('policy', 'time')
        url = request.json['url']
        
        if isinstance(provider, str):
            provider = [provider]

        shots = request.json.get('shots')
        if not shots:
            return "Shots must be specified", 400

        user = uuid.uuid4().int
        providers = {}
        try:
            fragment = urlparse(url).fragment
        except:
            return "Invalid URL", 400

        parsed_url = urlparse(url)
        if parsed_url.netloc != "algassert.com" or 'quirk' not in parsed_url.path:
            return "URL must come from quirk", 400

        circuit_str = fragment[len('circuit='):] if fragment.startswith('circuit=') else None
        if not circuit_str:
            return "Invalid URL", 400
        circuit = ast.literal_eval(unquote(circuit_str))

        for provider_name in provider:
            if provider_name == 'ibm':
                num_qubits = max(len(col) for col in circuit['cols'])
                providers['ibm'] = shots
            elif provider_name == 'aws':
                num_qubits = max(len(col) for col in circuit['cols'] if 'Measure' not in col)
                providers['aws'] = shots
            elif provider_name == 'microsoft':
                num_qubits = max(len(col) for col in circuit['cols'])
                providers['microsoft'] = shots

        for provider in providers:
            shots = providers[provider]
            maxDepth = max(sum(1 for j in circuit['cols'] if i < len(j) and j[i] not in {1, 'Measure'}) for i in range(num_qubits))
            self.select_policy(url, num_qubits, shots, user, url, maxDepth, provider, policy, 0)
        return str(user), 200

    def store_url_circuit(self) -> tuple:
        """
        Sends the GitHub URL of the circuit to the policy service.
        It first needs to get the content of the file, check if its a quantum circuit and parse it to a standard way.

        Request Parameters:
            url (str): The GitHub URL of the circuit
            shots (int): The number of shots to execute the circuit
            policy (str): The policy to execute the circuit. Default is 'time'

        Returns:
            tuple: The response of the policy service with the scheduler task identification
        """
        if request.json.get('url') is None:
            return "URL must be specified", 400
        if request.json.get('shots') is None:
            return "Shots must be specified", 400
        if request.json.get('policy') is None:
            policy = 'time'
        else:
            policy = request.json['policy']

        url = request.json['url']
        shots = request.json['shots']
        criterio = request.json['criterio']

        if not isinstance(shots, int) or shots <= 0 or shots > 20000:
            return "Invalid shots value", 400

        user = uuid.uuid4().int
        document = {
            '_id': str(user),
            'circuit': url
        }
        #with self.result_lock:
        #    self.collection.insert_one(document)

        try:
            parsed_url = urlparse(url)
            if parsed_url.netloc != "raw.githubusercontent.com":
                return "URL must come from a raw GitHub file", 400
            response = requests.get(url)
            response.raise_for_status()
            circuit_name = url.split('/')[-1]
        except requests.exceptions.RequestException as e:
            print(f"Error getting URL content: {e}")
            return "Invalid URL", 400
        
        circuit = response.text
        lines = circuit.split('\n')
        importAWS = next((line for line in lines if 'braket.circuits' in line), None)
        importIBM = next((line for line in lines if 'qiskit' in line), None)
        importMSFT = next((line for line in lines if 'microsoft' in line.lower() or 'qiskit' in line), None)  # <-- soporte Microsoft

        if importIBM:
            circ = self.executeCircuitIBM.code_to_circuit_ibm(circuit)
            num_qubits_line = next((line.split('#')[0].strip() for line in lines if '= QuantumRegister(' in line.split('#')[0]), None)
            num_qubits = int(num_qubits_line.split('QuantumRegister(')[1].split(',')[0].strip(')')) if num_qubits_line else None

            if num_qubits > self.scheduler_policies.getMaxQubits():
                return "Circuit too large", 400

            file_circuit_name_line = next((line.split('#')[0].strip() for line in lines if '= QuantumCircuit(' in line.split('#')[0]), None)
            file_circuit_name = file_circuit_name_line.split('=')[0].strip() if file_circuit_name_line else None

            qreg_line = next((line.split('#')[0].strip() for line in lines if '= QuantumRegister(' in line.split('#')[0]), None)
            qreg = qreg_line.split('=')[0].strip() if qreg_line else None
            creg_line = next((line.split('#')[0].strip() for line in lines if '= ClassicalRegister(' in line.split('#')[0]), None)
            creg = creg_line.split('=')[0].strip() if creg_line else None

            circuit_lines = [line.split('#')[0].strip() for line in lines if line.split('#')[0].strip().startswith(file_circuit_name+'.') and 'add_register' not in line]
            circuit = '\n'.join(circuit_lines)
            circuit = circuit.replace(file_circuit_name+'.', 'circuit.')
            circuit = circuit.replace(f'{qreg}[', 'qreg_q[')
            circuit = circuit.replace(f'{creg}[', 'creg_c[')

            qubits = [0] * num_qubits
            for line in circuit.split('\n'):
                if 'measure' not in line and 'barrier' not in line:
                    for match in re.finditer(r'qreg_q\[(\d+)\]', line):
                        qubits[int(match.group(1))] += 1
            maxDepth = max(qubits) if self.transpilation_machine == 'local' else 1
            provider = 'ibm'

        elif importAWS:
            file_circuit_name_line = next((line.split('#')[0].strip() for line in lines if '= Circuit(' in line.split('#')[0]), None)
            file_circuit_name = file_circuit_name_line.split('=')[0].strip() if file_circuit_name_line else None
            circuit_lines = [line.split('#')[0].strip() for line in lines if line.split('#')[0].strip().startswith(file_circuit_name+'.') and 'add_register' not in line]
            circuit = '\n'.join(circuit_lines)
            circuit = circuit.replace(file_circuit_name+'.', 'circuit.')
            circuit = '\n'.join([line.lstrip() for line in circuit.split('\n')])

            qubits = {}
            for line in circuit.split('\n'):
                if 'barrier' not in line and 'circuit.' in line:
                    gate_name = re.search(r'circuit\.(.*?)\(', line).group(1)
                    if gate_name in ['rx','ry','rz','gpi','gpi2','phaseshift']:
                        numbers_retrieved = re.findall(r'\d+', line)
                        numbers = numbers_retrieved[0] if numbers_retrieved else None
                    elif gate_name in ['xx','yy','zz','ms'] or 'cphase' in gate_name:
                        numbers_retrieved = re.findall(r'\d+', line)
                        numbers = numbers[:2] if numbers_retrieved else None
                    else:
                        numbers = re.findall(r'\d+', line)
                    for elem in numbers:
                        qubits[elem] = qubits.get(elem, 0) + 1
            maxDepth = max(qubits.values()) if self.transpilation_machine == 'local' else max(qubits.values())
            num_qubits = len(qubits.values())
            provider = 'aws'

        elif importMSFT:  # <-- nuevo bloque Microsoft
            circ = self.executeCircuitMicrosoft.code_to_circuit(circuit)
            num_qubits = circ.num_qubits
            qubits = [0] * num_qubits
            for line in circuit.split('\n'):
                for match in re.finditer(r'qreg_q\[(\d+)\]', line):
                    qubits[int(match.group(1))] += 1
            maxDepth = max(qubits) if self.transpilation_machine == 'local' else 1
            provider = 'microsoft'

        self.select_policy(circuit, num_qubits, shots, user, circuit_name, maxDepth, provider, policy, criterio)
        return str(user), 200


    def sendResults(self) -> tuple:
        id = request.args.get('id')
        if not id:
            return "No id provided", 400
        try:
            user = int(id)
        except:
            return "Invalid id value. It must be an integer.", 400
        if user <= 0:
            return "Invalid id value. It must be a positive integer.", 400
        cursor = self.collection.find({'_id': str(user)},{'_id': 0})
        documents = list(cursor)
        return json.dumps(dumps(documents)), 200

    def updatePorts(self) -> None:
        for i in range(8083, 8182):
            a_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            location = ("0.0.0.0", i)
            result_of_check = a_socket.connect_ex(location)
            self.ports[i] = 0 if result_of_check != 0 else 1
            a_socket.close()

    def getFreePort(self) -> int:
        puertos = [k for k, v in self.ports.items() if v == 0]
        self.ports[puertos[0]] = 1
        return puertos[0]

if __name__ == '__main__':
    app = Scheduler()
    app.run()
