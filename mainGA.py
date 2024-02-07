import sys
from PyQt5.QtWidgets import (
    QApplication,
    QWidget,
    QVBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QListView,
)
import numpy
import GA
from PyQt5.QtCore import QStringListModel


class GeneticAlgorithmUI(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle("Genetic Algorithm")
        self.setGeometry(100, 100, 500, 400)

        # Etiketler ve giriş kutuları
        self.equation_inputs_label = QLabel("Equation Inputs (comma separated): ")
        self.equation_inputs_input = QLineEdit(self)
        self.num_weights_label = QLabel("Number of Weights: ")
        self.num_weights_input = QLineEdit(self)
        self.sol_per_pop_label = QLabel("Solutions per Population: ")
        self.sol_per_pop_input = QLineEdit(self)
        self.low_label = QLabel("Low value: ")
        self.low_input = QLineEdit(self)
        self.high_label = QLabel("High value: ")
        self.high_input = QLineEdit(self)
        self.num_generations_label = QLabel("Number of Generations: ")
        self.num_generations_input = QLineEdit(self)
        self.num_parents_mating_label = QLabel("Number of Parents for Mating: ")
        self.num_parents_mating_input = QLineEdit(self)

        # GENERATE butonu
        self.generate_button = QPushButton("GENERATE", self)
        self.generate_button.clicked.connect(self.runGeneticAlgorithm)

        # ListView
        self.list_view = QListView()

        # Layout'lar
        vbox = QVBoxLayout()
        vbox.addWidget(self.equation_inputs_label)
        vbox.addWidget(self.equation_inputs_input)
        vbox.addWidget(self.num_weights_label)
        vbox.addWidget(self.num_weights_input)
        vbox.addWidget(self.sol_per_pop_label)
        vbox.addWidget(self.sol_per_pop_input)
        vbox.addWidget(self.low_label)
        vbox.addWidget(self.low_input)
        vbox.addWidget(self.high_label)
        vbox.addWidget(self.high_input)
        vbox.addWidget(self.num_generations_label)
        vbox.addWidget(self.num_generations_input)
        vbox.addWidget(self.num_parents_mating_label)
        vbox.addWidget(self.num_parents_mating_input)
        vbox.addWidget(self.generate_button)
        vbox.addWidget(self.list_view)

        self.setLayout(vbox)

    def runGeneticAlgorithm(self):
        try:
            equation_inputs = [
                float(x.strip()) for x in self.equation_inputs_input.text().split(",")
            ]
            num_weights = int(self.num_weights_input.text())
            sol_per_pop = int(self.sol_per_pop_input.text())
            low = float(self.low_input.text())
            high = float(self.high_input.text())
            num_generations = int(self.num_generations_input.text())
            num_parents_mating = int(self.num_parents_mating_input.text())

            pop_size = (sol_per_pop, num_weights)
            new_population = numpy.random.uniform(low=low, high=high, size=pop_size)

            result_list = []
            for generation in range(num_generations):
                fitness = GA.cal_pop_fitness(equation_inputs, new_population)
                parents = GA.select_mating_pool(
                    new_population, fitness, num_parents_mating
                )
                offspring_crossover = GA.crossover(
                    parents,
                    offspring_size=(pop_size[0] - parents.shape[0], num_weights),
                )
                offspring_mutation = GA.mutation(offspring_crossover)
                new_population[0 : parents.shape[0], :] = parents
                new_population[parents.shape[0] :, :] = offspring_mutation
                fitness = GA.cal_pop_fitness(equation_inputs, new_population)
                best_match_idx = numpy.where(fitness == numpy.max(fitness))

                # result_list.append(
                #    f"Generation {generation}: Best solution fitness: {fitness[best_match_idx]}"
                # ),
                result_list.append(
                    f"Generation {generation}: \n"
                    f"Best result: {numpy.max(numpy.sum(new_population*equation_inputs, axis=1))} \n"
                    f"Best solution: {new_population[best_match_idx, :]} \n"
                    f"Best solution fitness: {fitness[best_match_idx]} \n"
                    "==================================================="
                ),

            # Sonuçları ListView'a ekleme
            model = QStringListModel()
            model.setStringList(result_list)
            self.list_view.setModel(model)

        except Exception as e:
            print(f"An error occurred: {str(e)}")


def main():
    app = QApplication(sys.argv)
    ex = GeneticAlgorithmUI()
    # ex.show()
    ex.showMaximized()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
