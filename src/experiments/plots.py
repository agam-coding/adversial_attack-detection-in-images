import glob
import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

from src.experiments.stats import ExperimentStats


def show_save_fig(fig, show: bool, save_path: str = None):
    if save_path is not None:
        fig.savefig(save_path)
        print(f"Plot saved to {save_path}")
    if show:
        fig.show()


def latest_file_with_prefix(pattern: str):
    print(f"Using glob pattern: {pattern}")
    list_of_files = glob.glob(pattern)
    print(f"Matching files: {list_of_files}")
    if not list_of_files:
        print(f"No files found matching pattern: {pattern}")
        return None
    latest_file = max(list_of_files, key=os.path.getctime)
    print(f"Selected latest file: {latest_file}")
    return latest_file


def get_stats(pattern: str):
    file_path = latest_file_with_prefix(pattern)
    if file_path is None:
        return None
    stats = ExperimentStats()
    try:
        stats.load(file_path)
        print(f"Loaded {len(stats.records)} records from {file_path}")
        if stats.records:
            # Print info about the first record
            record = stats.records[0]
            print(f"First record has {len(record.results)} results")
            if record.results:
                fitness = record.results[0].best_individuals_fitness
                print(f"First result fitness length: {len(fitness) if fitness else 'empty'}")
        return stats
    except Exception as e:
        print(f"Error loading stats from {file_path}: {e}")
        return None


def boxplot(pattern: str, save_path: str):
    stats = get_stats(pattern)
    if stats is None:
        print(f"Skipping boxplot for {pattern} as no data is available")
        return
        
    records_dict = {}
    # for each record, extract fitness of each result
    for record in stats.records:
        fitnesses = [result.best_individuals_fitness[-1] for result in record.results if result.best_individuals_fitness]
        if not fitnesses:
            continue

        label = F"(CP={record.params.crossover_probability}, " \
                F"MP={record.params.mutation_probability}, " \
                F"NG={record.params.mutation_num_genes})"
        records_dict[label] = fitnesses

    if not records_dict:
        print(f"No valid fitness data found for boxplot")
        return
    
    # Create a more visually appealing boxplot with seaborn
    plt.figure(figsize=(12, 8))
    
    # Convert data to format for seaborn
    all_data = []
    for label, values in records_dict.items():
        for val in values:
            all_data.append({'Parameter Set': label, 'Fitness': val})
    
    # Create DataFrame
    import pandas as pd
    df = pd.DataFrame(all_data)
    
    # Create boxplot
    ax = sns.boxplot(x='Parameter Set', y='Fitness', data=df, palette='viridis')
    ax.set_xlabel("Parameters", fontsize=12)
    ax.set_ylabel("Fitness", fontsize=12)
    ax.set_title("Fitness Distribution by Parameter Set", fontsize=14)
    plt.xticks(rotation=90)
    plt.tight_layout()
    
    show_save_fig(plt.gcf(), show=True, save_path=save_path)


def boxplot_popsize(pattern: str, save_path: str):
    stats = get_stats(pattern)
    if stats is None:
        print(f"Skipping boxplot_popsize for {pattern} as no data is available")
        return
        
    records_dict = {}
    # for each record, extract fitness of each result
    for record in stats.records:
        fitnesses = [result.best_individuals_fitness[-1] for result in record.results if result.best_individuals_fitness]
        if not fitnesses:
            continue
            
        popsize = record.params.population_size
        records_dict[popsize] = fitnesses

    if not records_dict:
        print(f"No valid fitness data found for boxplot_popsize")
        return
    
    # Create a more visually appealing boxplot with seaborn
    plt.figure(figsize=(12, 8))
    
    # Convert data to format for seaborn
    all_data = []
    for popsize, values in records_dict.items():
        for val in values:
            all_data.append({'Population Size': popsize, 'Fitness': val})
    
    # Create DataFrame
    import pandas as pd
    df = pd.DataFrame(all_data)
    
    # Create boxplot
    ax = sns.boxplot(x='Population Size', y='Fitness', data=df, palette='viridis')
    ax.set_xlabel("Population Size", fontsize=12)
    ax.set_ylabel("Fitness", fontsize=12)
    ax.set_title("Fitness Distribution by Population Size", fontsize=14)
    plt.tight_layout()
    
    show_save_fig(plt.gcf(), show=True, save_path=save_path)


def plot_fitness_over_generations(pattern: str, save_path: str):
    stats = get_stats(pattern)
    if stats is None:
        print(f"Skipping plot_fitness_over_generations for {pattern} as no data is available")
        return
        
    try:
        if len(stats.records) > 1:
            raise ValueError("There should be only a single record")
        
        if not stats.records:
            print("No records found in the stats")
            return
            
        record = stats.records[0]
        
        if not record.results:
            print("No results found in the record")
            return
        
        print(f"Record params: generations={record.params.generations}")
        
        fitnesses = [result.best_individuals_fitness for result in record.results]
        print(f"Fitnesses shapes: {[len(f) if f else 'empty' for f in fitnesses]}")
        
        # Check for empty fitnesses
        if not all(fitnesses):
            print("Some fitness arrays are empty")
            return
        
        # Make sure all fitness arrays are the same length
        min_length = min(len(f) for f in fitnesses)
        fitnesses = [f[:min_length] for f in fitnesses]
        
        fitnesses = np.array(fitnesses)
        print(f"Fitnesses array shape: {fitnesses.shape}")
        
        fitnesses_mean = np.mean(fitnesses, axis=0)
        print(f"Mean fitnesses shape: {fitnesses_mean.shape}")
        
        generations = np.arange(0, min_length)
        print(f"Generations shape: {generations.shape}")
        
        # Create an enhanced plot with confidence intervals
        plt.figure(figsize=(12, 8))
        
        # Calculate 95% confidence interval
        fitnesses_std = np.std(fitnesses, axis=0)
        conf_interval = 1.96 * fitnesses_std / np.sqrt(len(fitnesses))
        
        plt.plot(generations, fitnesses_mean, 'b-', linewidth=2, label='Mean Fitness')
        plt.fill_between(generations, 
                         fitnesses_mean - conf_interval, 
                         fitnesses_mean + conf_interval, 
                         color='blue', alpha=0.2, label='95% Confidence Interval')
        
        # Add individual fitness traces with low opacity
        for i, fitness in enumerate(fitnesses):
            plt.plot(generations, fitness, 'b-', alpha=0.2, linewidth=0.5, label=f'Run {i+1}' if i==0 else None)
        
        plt.xlabel("Generations", fontsize=12)
        plt.ylabel("Fitness", fontsize=12)
        plt.title("Fitness Evolution over Generations", fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        plt.tight_layout()
        
        show_save_fig(plt.gcf(), show=True, save_path=save_path)
        
    except Exception as e:
        print(f"Error in plot_fitness_over_generations: {e}")
        import traceback
        traceback.print_exc()


def create_parameter_heatmap(pattern: str, save_path: str, param1_name: str, param2_name: str):
    """Create a heatmap showing how two parameters affect the fitness."""
    stats = get_stats(pattern)
    if stats is None:
        print(f"Skipping heatmap for {pattern} as no data is available")
        return
    
    try:
        # Extract unique values for the parameters
        param1_values = set()
        param2_values = set()
        
        for record in stats.records:
            param1 = getattr(record.params, param1_name, None)
            param2 = getattr(record.params, param2_name, None)
            
            if param1 is not None and param2 is not None:
                param1_values.add(param1)
                param2_values.add(param2)
        
        param1_values = sorted(list(param1_values))
        param2_values = sorted(list(param2_values))
        
        if not param1_values or not param2_values:
            print(f"No valid parameter values found for heatmap")
            return
        
        # Create a matrix for the heatmap
        heatmap_data = np.zeros((len(param1_values), len(param2_values)))
        
        # Fill the matrix with average fitness values
        for i, param1 in enumerate(param1_values):
            for j, param2 in enumerate(param2_values):
                # Find corresponding records
                records = [r for r in stats.records 
                          if getattr(r.params, param1_name, None) == param1 
                          and getattr(r.params, param2_name, None) == param2]
                
                if records:
                    # Average the final fitness across all matching records and repetitions
                    all_fitnesses = []
                    for record in records:
                        for result in record.results:
                            if result.best_individuals_fitness:
                                all_fitnesses.append(result.best_individuals_fitness[-1])
                    
                    if all_fitnesses:
                        heatmap_data[i, j] = np.mean(all_fitnesses)
        
        # Create a fancy heatmap
        plt.figure(figsize=(12, 10))
        
        # Create a nicer colormap
        colors = ["#053061", "#2166AC", "#4393C3", "#92C5DE", "#D1E5F0", 
                 "#F7F7F7", "#FDDBC7", "#F4A582", "#D6604D", "#B2182B", "#67001F"]
        cmap = LinearSegmentedColormap.from_list("custom_cmap", colors, N=256)
        
        ax = sns.heatmap(heatmap_data, 
                        xticklabels=param2_values,
                        yticklabels=param1_values,
                        cmap=cmap,
                        annot=True,  # Show values
                        fmt=".3f",   # Format for the values
                        cbar_kws={'label': 'Average Fitness'})
        
        plt.xlabel(param2_name, fontsize=12)
        plt.ylabel(param1_name, fontsize=12)
        plt.title(f"Average Fitness by {param1_name} and {param2_name}", fontsize=14)
        plt.tight_layout()
        
        show_save_fig(plt.gcf(), show=True, save_path=save_path)
        
    except Exception as e:
        print(f"Error creating heatmap: {e}")
        import traceback
        traceback.print_exc()


def parameter_line_plot(pattern: str, save_path: str, param_name: str):
    """Create a line plot showing how a single parameter affects fitness."""
    stats = get_stats(pattern)
    if stats is None:
        print(f"Skipping line plot for {pattern} as no data is available")
        return
    
    try:
        # Extract data for the parameter
        param_values = []
        fitness_means = []
        fitness_stds = []
        
        # Group records by parameter value
        param_groups = {}
        for record in stats.records:
            param_value = getattr(record.params, param_name, None)
            if param_value is None:
                continue
                
            if param_value not in param_groups:
                param_groups[param_value] = []
                
            # Extract final fitness values
            for result in record.results:
                if result.best_individuals_fitness:
                    param_groups[param_value].append(result.best_individuals_fitness[-1])
        
        # Calculate statistics for each parameter value
        for param_value, fitnesses in sorted(param_groups.items()):
            if fitnesses:
                param_values.append(param_value)
                fitness_means.append(np.mean(fitnesses))
                fitness_stds.append(np.std(fitnesses))
        
        if not param_values:
            print(f"No valid parameter values found for line plot")
            return
        
        # Create an enhanced line plot
        plt.figure(figsize=(12, 8))
        
        plt.errorbar(param_values, fitness_means, yerr=fitness_stds, 
                    fmt='o-', ecolor='gray', capsize=5, linewidth=2, markersize=8)
        
        plt.xlabel(param_name, fontsize=12)
        plt.ylabel("Average Fitness", fontsize=12)
        plt.title(f"Effect of {param_name} on Fitness", fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Add values on the plot
        for i, (x, y) in enumerate(zip(param_values, fitness_means)):
            plt.annotate(f"{y:.3f}", (x, y), textcoords="offset points", 
                        xytext=(0,10), ha='center', fontsize=9)
        
        plt.tight_layout()
        
        show_save_fig(plt.gcf(), show=True, save_path=save_path)
        
    except Exception as e:
        print(f"Error creating line plot: {e}")
        import traceback
        traceback.print_exc()


def save_experiment_results_text(pattern: str, output_file: str):
    """
    Save the results of an experiment to a text file for easier examination.
    """
    stats = get_stats(pattern)
    if stats is None:
        print(f"No data available for {pattern}")
        return
    
    try:
        with open(output_file, 'w') as f:
            f.write(f"Experiment Results for {pattern}\n")
            f.write("=" * 50 + "\n\n")
            
            for i, record in enumerate(stats.records):
                f.write(f"Record {i+1}:\n")
                f.write(f"  Parameters:\n")
                f.write(f"    generations: {record.params.generations}\n")
                f.write(f"    population_size: {record.params.population_size}\n")
                f.write(f"    mutation_probability: {record.params.mutation_probability}\n")
                f.write(f"    mutation_num_genes: {record.params.mutation_num_genes}\n")
                f.write(f"    crossover_probability: {record.params.crossover_probability}\n")
                f.write(f"    perturbation_importance: {record.params.perturbation_importance}\n")
                
                f.write(f"  Results ({len(record.results)} repetitions):\n")
                for j, result in enumerate(record.results):
                    f.write(f"    Repetition {j+1}:\n")
                    if result.best_individuals_fitness:
                        f.write(f"      Final fitness: {result.best_individuals_fitness[-1]}\n")
                        f.write(f"      Initial fitness: {result.best_individuals_fitness[0]}\n")
                        f.write(f"      Improvement: {result.best_individuals_fitness[-1] - result.best_individuals_fitness[0]}\n")
                    else:
                        f.write(f"      No fitness data available\n")
                    f.write(f"      Accuracy: {result.accuracy}\n")
                
                # Calculate average results
                if record.results:
                    avg_accuracy = sum(r.accuracy for r in record.results) / len(record.results)
                    final_fitnesses = [r.best_individuals_fitness[-1] if r.best_individuals_fitness else 0 for r in record.results]
                    if final_fitnesses:
                        avg_final_fitness = sum(final_fitnesses) / len(final_fitnesses)
                        f.write(f"  Average Results:\n")
                        f.write(f"    Average accuracy: {avg_accuracy}\n")
                        f.write(f"    Average final fitness: {avg_final_fitness}\n")
                
                f.write("\n" + "-" * 40 + "\n\n")
            
            f.write("\nAnalysis completed at " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        
        print(f"***************Results saved to {output_file}***************")
    except Exception as e:
        print(f"Error saving results to text file: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # For generations experiment
    plot_fitness_over_generations('experiments/generations*', 'docs/generations_line.png')
    save_experiment_results_text('experiments/generations*', 'docs/generations_results.txt')
    
    # For population size experiment
    boxplot_popsize('experiments/popsize*', 'docs/popsize_boxplot.png')
    parameter_line_plot('experiments/popsize*', 'docs/popsize_line.png', 'population_size')
    save_experiment_results_text('experiments/popsize*', 'docs/popsize_results.txt')
    
    # For mutation probability experiment
    boxplot('experiments/mutation_prob*', 'docs/mutation_prob_boxplot.png')
    parameter_line_plot('experiments/mutation_prob*', 'docs/mutation_prob_line.png', 'mutation_probability')
    save_experiment_results_text('experiments/mutation_prob*', 'docs/mutation_prob_results.txt')
    
    # For mutation number of genes experiment
    boxplot('experiments/mutation_numgenes*', 'docs/mutation_numgenes_boxplot.png')
    parameter_line_plot('experiments/mutation_numgenes*', 'docs/mutation_numgenes_line.png', 'mutation_num_genes')
    save_experiment_results_text('experiments/mutation_numgenes*', 'docs/mutation_numgenes_results.txt')
    
    # For crossover probability experiment
    boxplot('experiments/crossover_prob*', 'docs/crossover_prob_boxplot.png')
    parameter_line_plot('experiments/crossover_prob*', 'docs/crossover_prob_line.png', 'crossover_probability')
    save_experiment_results_text('experiments/crossover_prob*', 'docs/crossover_prob_results.txt')
    
    # Create heatmaps for combinations of parameters if enough data
    create_parameter_heatmap('experiments/mutation_numgenes*', 'docs/heatmap_mutation_genes_vs_popsize.png', 
                           'mutation_num_genes', 'population_size')
