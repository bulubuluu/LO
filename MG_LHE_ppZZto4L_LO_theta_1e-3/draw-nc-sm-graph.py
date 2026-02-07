"""
This script reads event data from ROOT files, calculates weights based on SM/NC amplitudes,
generates 1D/2D plots for Z boson kinematic analysis, and exports extreme event data to PDF.

Key Fixes & Features:
1. Correct Z boson counting per event (ak.sum(z_mask, axis=1) instead of ak.num(z_mask))
2. Automatic axis range calculation using percentiles (0.1%/99.9%) to filter outliers
3. Fix syntax error: variable name cannot start with number (2d_plot_configs → plot_2d_configs)
4. Fix AxisError/AttributeError: replace ak.repeat (v2.x only) with numpy.repeat (compatible with all versions)
5. Critical fix: Use axis=None for ak.flatten to fully flatten Z boson arrays
"""

import uproot
import awkward as ak
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import cm


def get_auto_range(data, percentile_low=0.1, percentile_high=99.9, show_all=False):
    """
    Calculate axis range automatically for plot visualization (filter outliers by default)
    Args:
        data: 1D numpy array of plot data
        percentile_low: Lower percentile to filter outliers
        percentile_high: Upper percentile to filter outliers
        show_all: If True, use min/max to show all data points; else use percentiles (default)
    Returns:
        Tuple (min_val, max_val) for axis range
    """
    if len(data) == 0 or not np.any(np.isfinite(data)):
        return (0, 100)
    
    finite_data = data[np.isfinite(data)]
    if show_all:
        min_val = np.min(finite_data)
        max_val = np.max(finite_data)
    else:
        min_val = np.percentile(finite_data, percentile_low)
        max_val = np.percentile(finite_data, percentile_high)
        if np.isclose(min_val, max_val):
            min_val -= 0.1 * abs(min_val) if min_val != 0 else -1
            max_val += 0.1 * abs(max_val) if max_val != 0 else 1
    return (min_val, max_val)


def plot_2d_distribution(x_data, y_data, weights, title, xlabel, ylabel, filename,
                         bins=(50, 50), range=None, cmap='viridis'):
    """
    Generate 2D histogram plot with automatic axis range (filter outliers)
    Args:
        x_data: X-axis data array (1D numpy)
        y_data: Y-axis data array (1D numpy)
        weights: Weight array for weighted plot (None for non-weighted)
        title: Plot title string
        xlabel: X-axis label string
        ylabel: Y-axis label string
        filename: Output file path
        bins: Number of bins for histogram (tuple for x/y)
        range: Predefined axis range ((x_min,x_max), (y_min,y_max))
        cmap: Matplotlib colormap name
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Calculate axis range with outlier filtering if not provided
    if range is None:
        x_range = get_auto_range(x_data, show_all=False)
        y_range = get_auto_range(y_data, show_all=False)
        range = (x_range, y_range)
    
    # Filter non-finite values
    finite_mask = np.isfinite(x_data) & np.isfinite(y_data)
    if weights is not None:
        finite_mask &= np.isfinite(weights)
        plot_weights = weights[finite_mask]
    else:
        plot_weights = None
    
    x_finite = x_data[finite_mask]
    y_finite = y_data[finite_mask]
    
    # Create 2D histogram
    hist, xedges, yedges, im = ax.hist2d(
        x_finite, y_finite, bins=bins, range=range, weights=plot_weights, cmap=cmap
    )
    plt.colorbar(im, ax=ax, label='Events')
    
    # Plot styling
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.grid(alpha=0.3, linestyle='--')
    
    # Save and close plot
    fig.tight_layout()
    fig.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close(fig)
    return fig


def plot_weighted_comparison(original_values, weighted_values, weights, title, xlabel, filename,
                             bins=100, range=None, label1="Original", label2="Weighted", 
                             color1='blue', color2='red', normalize=False,
                             values3=None, weights3=None, label3=None, color3='green'):
    """
    Generate 1D weighted comparison histogram (filter outliers by default)
    Args:
        original_values: Unweighted data array
        weighted_values: Data array for weighted comparison
        weights: Weight array for weighted plot
        title: Plot title string
        xlabel: X-axis label string
        filename: Output file path
        bins: Number of histogram bins
        range: Predefined X-axis range
        label1: Legend label for original data
        label2: Legend label for weighted data
        color1: Color for original data plot
        color2: Color for weighted data plot
        normalize: If True, normalize histogram to probability density
        values3: Optional third data array for comparison
        weights3: Optional weights for third data array
        label3: Legend label for third data array
        color3: Color for third data array plot
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    # Calculate axis range with outlier filtering if not provided
    if range is None:
        combined = []
        if len(original_values) > 0:
            combined.append(original_values[np.isfinite(original_values)])
        if len(weighted_values) > 0:
            combined.append(weighted_values[np.isfinite(weighted_values)])
        if values3 is not None and len(values3) > 0:
            combined.append(values3[np.isfinite(values3)])
        combined = np.concatenate(combined) if combined else np.array([])
        range = get_auto_range(combined, show_all=False)

    # Filter non-finite values
    orig_finite = original_values[np.isfinite(original_values)]
    wgt_finite = weighted_values[np.isfinite(weighted_values)]
    wgt_weights = weights[np.isfinite(weights)] if weights is not None else None

    # Plot original data
    ax.hist(orig_finite, bins=bins, range=range, histtype='step',
            linewidth=2, color=color1, alpha=0.8, label=label1, density=normalize)
    
    # Plot weighted data
    if wgt_weights is not None:
        ax.hist(wgt_finite, bins=bins, range=range, histtype='step',
                linewidth=2, color=color2, alpha=0.8, label=label2, density=normalize,
                weights=wgt_weights)
    
    # Plot optional third data set
    if values3 is not None and weights3 is not None and label3 is not None:
        val3_finite = values3[np.isfinite(values3)]
        wgt3_finite = weights3[np.isfinite(weights3)]
        ax.hist(val3_finite, bins=bins, range=range, histtype='step',
                linewidth=2, color=color3, alpha=0.8, label=label3, density=normalize,
                weights=wgt3_finite)

    # Plot styling
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel('Events', fontsize=12)
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(alpha=0.3)

    # Save and close plot
    fig.tight_layout()
    fig.savefig(filename, dpi=300)
    plt.close(fig)
    return fig


def plot_amplitude_scatter(sm_values, nc_values, title, filename, xlabel="SM Amplitude", ylabel="NC Amplitude",
                           alpha=0.5, s=10, xlim=None, ylim=None):
    """
    Generate scatter plot for SM vs NC amplitude (filter outliers by default)
    Args:
        sm_values: SM amplitude data array
        nc_values: NC amplitude data array
        title: Plot title string
        filename: Output file path
        xlabel: X-axis label string
        ylabel: Y-axis label string
        alpha: Scatter point transparency
        s: Scatter point size
        xlim: Predefined X-axis range
        ylim: Predefined Y-axis range
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    # Calculate axis range with outlier filtering if not provided
    if xlim is None:
        xlim = get_auto_range(sm_values, show_all=False)
    if ylim is None:
        ylim = get_auto_range(nc_values, show_all=False)

    # Filter non-finite values
    finite_mask = np.isfinite(sm_values) & np.isfinite(nc_values)
    ax.scatter(sm_values[finite_mask], nc_values[finite_mask], alpha=alpha, s=s, color='purple')

    # Plot styling
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.grid(alpha=0.3)

    # Save and close plot
    fig.tight_layout()
    fig.savefig(filename, dpi=300)
    plt.close(fig)
    return fig


def plot_weight_distribution(weights, title, filename, bins=100, color='orange', range=None, normalize=False):
    """
    Generate 1D histogram for weight distribution (filter outliers by default)
    Args:
        weights: Weight data array
        title: Plot title string
        filename: Output file path
        bins: Number of histogram bins
        color: Histogram color
        range: Predefined X-axis range
        normalize: If True, normalize to probability density
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    # Calculate axis range with outlier filtering if not provided
    if range is None:
        range = get_auto_range(weights, show_all=False)
    
    # Filter non-finite values
    valid_weights = weights[np.isfinite(weights)]
    ax.hist(valid_weights, bins=bins, color=color, alpha=0.8, density=normalize, range=range)

    # Plot styling
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel("Weight Value (1 + NC/SM)", fontsize=12)
    ax.set_ylabel('Events', fontsize=12)
    ax.grid(alpha=0.3)

    # Save and close plot
    fig.tight_layout()
    fig.savefig(filename, dpi=300)
    plt.close(fig)
    return fig


def plot_weight_1d_comparison(weight_vals, title, filename, bins=100, range=None, 
                              color1='blue', color2='red', 
                              label1='SM (Non-weighted)', label2='BSM (Weighted)'):
    """
    Generate 1D comparison plot for weight distribution (1 + NC/SM)
    Args:
        weight_vals: Weight data array
        title: Plot title string
        filename: Output file path
        bins: Number of histogram bins
        range: Predefined X-axis range
        color1: Color for non-weighted plot
        color2: Color for weighted plot
        label1: Legend label for non-weighted data
        label2: Legend label for weighted data
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Filter non-finite values
    finite_mask = np.isfinite(weight_vals)
    weights_finite = weight_vals[finite_mask]
    
    # Calculate axis range with outlier filtering if not provided
    if range is None:
        range = get_auto_range(weights_finite, show_all=False)

    # Plot non-weighted and weighted histograms
    ax.hist(weights_finite, bins=bins, range=range, histtype='step', linewidth=2, 
            color=color1, label=label1, density=False)
    
    ax.hist(weights_finite, bins=bins, range=range, histtype='step', linewidth=2, 
            color=color2, label=label2, density=False, weights=weights_finite)
    
    # Plot styling
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('1 + NC/SM', fontsize=12)
    ax.set_ylabel('Events', fontsize=12)
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(alpha=0.3)
    
    # Save and close plot
    fig.tight_layout()
    fig.savefig(filename, dpi=300)
    plt.close(fig)
    return fig


def export_extreme_z_quark_momenta_to_pdf(extreme_events, extreme_weights, extreme_indices, output_dir):
    """
    Export extreme event data (1 + NC/SM > 500) to PDF report
    Args:
        extreme_events: Awkward array of extreme events
        extreme_weights: Weight array for extreme events
        extreme_indices: Indices of extreme events in original data
        output_dir: Directory to save PDF file
    """
    pdf_filename = os.path.join(output_dir, "extreme_events_z_quark_momenta.pdf")
    doc = SimpleDocTemplate(
        pdf_filename,
        pagesize=A4,
        rightMargin=2*cm,
        leftMargin=2*cm,
        topMargin=2*cm,
        bottomMargin=2*cm
    )
    elements = []
    styles = getSampleStyleSheet()

    # Define custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=16,
        spaceAfter=20,
        alignment=1,
        textColor=colors.darkblue
    )
    section_style = ParagraphStyle(
        'CustomSection',
        parent=styles['Heading2'],
        fontSize=14,
        spaceAfter=15,
        textColor=colors.darkred
    )
    content_style = ParagraphStyle(
        'CustomContent',
        parent=styles['Normal'],
        fontSize=11,
        spaceAfter=10,
        leading=14
    )

    # Main title
    main_title = Paragraph("Extreme Events (1 + NC/SM > 500) - Z Boson & Quark 4-Momentum", title_style)
    elements.append(main_title)
    elements.append(Spacer(1, 12))

    # Event statistics
    n_extreme = len(extreme_events)
    stats_text = f"Total extreme events found: {n_extreme}\nWeight range: {np.min(extreme_weights):.2f} ~ {np.max(extreme_weights):.2f}"
    stats = Paragraph(stats_text, content_style)
    elements.append(stats)
    elements.append(Spacer(1, 15))

    # Z Boson data table
    z_title = Paragraph("1. Z Boson Data (PID = 23, Status = 2)", section_style)
    elements.append(z_title)
    z_headers = [
        "Event Index", "Z Boson Index", "PID", "Status",
        "E (GeV)", "Px (GeV)", "Py (GeV)", "Pz (GeV)", "Weight (1+NC/SM)"
    ]
    z_table_data = [z_headers]

    # Quark data table
    quark_title = Paragraph("2. Quark Data (PID = ±1, ±2)", section_style)
    elements.append(quark_title)
    quark_headers = [
        "Event Index", "Quark Index", "PID (Quark Type)", "Status",
        "E (GeV)", "Px (GeV)", "Py (GeV)", "Pz (GeV)", "Weight (1+NC/SM)"
    ]
    quark_table_data = [quark_headers]

    # Populate tables with event data
    for event_idx, (event, weight, global_idx) in enumerate(zip(extreme_events, extreme_weights, extreme_indices)):
        # Process Z boson data
        z_mask = (event["Particle.PID"] == 23) & (event["Particle.Status"] == 2)
        if ak.sum(z_mask) > 0:
            z_particles = ak.zip({
                "PID": event["Particle.PID"][z_mask],
                "Status": event["Particle.Status"][z_mask],
                "E": event["Particle.E"][z_mask],
                "Px": event["Particle.Px"][z_mask],
                "Py": event["Particle.Py"][z_mask],
                "Pz": event["Particle.Pz"][z_mask]
            })
            z_np = ak.to_numpy(ak.fill_none(z_particles, np.nan))
            for z_idx, z in enumerate(z_np):
                z_table_data.append([
                    str(global_idx), str(z_idx),
                    str(int(z["PID"])) if np.isfinite(z["PID"]) else "NaN",
                    str(int(z["Status"])) if np.isfinite(z["Status"]) else "NaN",
                    f"{z['E']:.4f}" if np.isfinite(z["E"]) else "NaN",
                    f"{z['Px']:.4f}" if np.isfinite(z["Px"]) else "NaN",
                    f"{z['Py']:.4f}" if np.isfinite(z["Py"]) else "NaN",
                    f"{z['Pz']:.4f}" if np.isfinite(z["Pz"]) else "NaN",
                    f"{weight:.4f}"
                ])

        # Process quark data
        quark_pids = {1, -1, 2, -2}
        quark_mask = ak.any([event["Particle.PID"] == pid for pid in quark_pids], axis=0)
        if ak.sum(quark_mask) > 0:
            quark_particles = ak.zip({
                "PID": event["Particle.PID"][quark_mask],
                "Status": event["Particle.Status"][quark_mask],
                "E": event["Particle.E"][quark_mask],
                "Px": event["Particle.Px"][quark_mask],
                "Py": event["Particle.Py"][quark_mask],
                "Pz": event["Particle.Pz"][quark_mask]
            })
            quark_np = ak.to_numpy(ak.fill_none(quark_particles, np.nan))
            for q_idx, q in enumerate(quark_np):
                pid_to_quark = {1: "d", -1: "anti-d", 2: "u", -2: "anti-u"}
                quark_type = pid_to_quark.get(int(q["PID"]), f"PID={int(q['PID'])}") if np.isfinite(q["PID"]) else "NaN"
                quark_table_data.append([
                    str(global_idx), str(q_idx),
                    quark_type,
                    str(int(q["Status"])) if np.isfinite(q["Status"]) else "NaN",
                    f"{q['E']:.4f}" if np.isfinite(q["E"]) else "NaN",
                    f"{q['Px']:.4f}" if np.isfinite(q["Px"]) else "NaN",
                    f"{q['Py']:.4f}" if np.isfinite(q["Py"]) else "NaN",
                    f"{q['Pz']:.4f}" if np.isfinite(q["Pz"]) else "NaN",
                    f"{weight:.4f}"
                ])

    # Add Z boson table to PDF
    if len(z_table_data) > 1:
        z_table = Table(z_table_data, repeatRows=1)
        z_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.lightblue),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('FONTSIZE', (0, 1), (-1, -1), 9),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey])
        ]))
        elements.append(z_table)
    else:
        elements.append(Paragraph("No Z boson data available for extreme events.", content_style))
    elements.append(Spacer(1, 20))

    # Add quark table to PDF
    if len(quark_table_data) > 1:
        quark_table = Table(quark_table_data, repeatRows=1)
        quark_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.lightgreen),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('FONTSIZE', (0, 1), (-1, -1), 9),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey])
        ]))
        elements.append(quark_table)
    else:
        elements.append(Paragraph("No quark data available for extreme events.", content_style))

    # Build PDF document
    doc.build(elements)
    print(f"Extreme events (Z + Quark) momentum PDF saved to: {pdf_filename}")


def main(input_file, tree_name, output_dir='./plots'):
    """
    Main function to process ROOT file, generate plots and export PDF
    Args:
        input_file: Path to input ROOT file
        tree_name: Name of the tree in ROOT file
        output_dir: Directory to save output plots/PDF
    """
    try:
        # Create output directory if not exists
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Created output directory: {output_dir}")

        # Define branches to read from ROOT file
        branches = [
            "Particle.PID", "Particle.Status", "Particle.M", "Particle.PT", "Particle.Eta", "Particle.Phi",
            "Particle.E", "Particle.Px", "Particle.Py", "Particle.Pz",
            "SM_Amplitude", "NC_Amplitude"
        ]
        print(f"Reading tree '{tree_name}' from file: {input_file}")
        
        # Read ROOT file data
        with uproot.open(input_file) as file:
            if tree_name not in file:
                raise KeyError(f"Tree '{tree_name}' not found in {input_file}")
            tree = file[tree_name]
            data = tree.arrays(branches, library="ak")
        print(f"Successfully loaded {len(data)} events")

        # Extract amplitude values and calculate weights
        sm_values = ak.to_numpy(data["SM_Amplitude"])
        nc_values = ak.to_numpy(data["NC_Amplitude"])
        valid_sm_mask = (sm_values != 0) & np.isfinite(sm_values) & np.isfinite(nc_values)
        print(f"Valid events (SM != 0 and finite): {np.sum(valid_sm_mask)}/{len(sm_values)}")
        
        new_weights = np.full_like(sm_values, np.nan, dtype=float)
        new_weights[valid_sm_mask] = 1 + (nc_values[valid_sm_mask] / sm_values[valid_sm_mask])
        print("Weight calculation completed: Weight = 1 + NC/SM")

        # Generate 1D weight comparison plot
        plot_weight_1d_comparison(
            weight_vals=new_weights,
            title="Distribution of 1 + NC/SM (SM vs BSM Events)",
            filename=f"{output_dir}/1+NC_SM_SM_vs_BSM.png",
            bins=100,
            range=None
        )
        print("1D comparison plot of (1+NC/SM) generated")

        # Generate amplitude comparison plot
        plot_weighted_comparison(
            original_values=sm_values[valid_sm_mask],
            weighted_values=nc_values[valid_sm_mask],
            weights=np.ones_like(nc_values[valid_sm_mask]),
            values3=sm_values[valid_sm_mask],
            weights3=new_weights[valid_sm_mask],
            label3="(1 + NC/SM)",
            color3='green',
            title="SM vs NC Amplitude",
            xlabel="Amplitude Value",
            filename=f"{output_dir}/amplitude_comparison.png",
            bins=100,
            range=None,
            label1="SM Amplitude",
            label2="NC Amplitude",
            color1='blue',
            color2='red',
            normalize=False
        )
        print("Amplitude comparison plot generated")

        # Generate weight distribution plot
        plot_weight_distribution(
            weights=new_weights,
            title="Weight Distribution (1 + NC/SM) - Valid Events",
            filename=f"{output_dir}/weight_distribution_1+NC_SM.png",
            bins=100,
            color='darkorange',
            range=None,
            normalize=False
        )
        print("Weight distribution plot generated")

        # Generate SM vs NC amplitude scatter plot
        plot_amplitude_scatter(
            sm_values=sm_values[valid_sm_mask],
            nc_values=nc_values[valid_sm_mask],
            title="SM Amplitude vs NC Amplitude (Valid Events)",
            filename=f"{output_dir}/sm_vs_nc_amplitude_scatter.png",
            alpha=0.5,
            s=5,
            xlim=None,
            ylim=None
        )
        print("SM-NC amplitude scatter plot generated")

        # Process Z boson data (fixed counting logic)
        z_mask = (data["Particle.PID"] == 23) & (data["Particle.Status"] == 2)
        total_z_particles = ak.sum(z_mask)
        print(f"Total Z bosons in all events: {total_z_particles}")
        if total_z_particles == 0:
            raise ValueError("No Z bosons found (PID=23, Status=2)")
        
        # Flatten Z boson kinematic variables (axis=None to get full 1D array)
        z_masses = ak.to_numpy(ak.flatten(data["Particle.M"][z_mask], axis=None))
        z_pt = ak.to_numpy(ak.flatten(data["Particle.PT"][z_mask], axis=None))
        z_eta = ak.to_numpy(ak.flatten(data["Particle.Eta"][z_mask], axis=None))
        z_phi = ak.to_numpy(ak.flatten(data["Particle.Phi"][z_mask], axis=None))
        z_energy = ak.to_numpy(ak.flatten(data["Particle.E"][z_mask], axis=None))
        z_pz = ak.to_numpy(ak.flatten(data["Particle.Pz"][z_mask], axis=None))
        
        # Correctly count Z bosons per event (sum over particle axis)
        z_count_per_event = ak.sum(z_mask, axis=1)
        z_count_np = ak.to_numpy(z_count_per_event)
        sum_z_count = np.sum(z_count_np)
        
        # Validate Z boson count consistency
        assert sum_z_count == total_z_particles, f"Sum of Z per event ({sum_z_count}) != total Z ({total_z_particles})"
        assert len(z_masses) == total_z_particles, f"Z masses length ({len(z_masses)}) != total Z ({total_z_particles})"
        
        # Repeat weights to match Z boson count per event
        z_new_weights = np.repeat(new_weights, z_count_np)
        
        # Final validation of weight-Z boson length match
        assert len(z_masses) == len(z_new_weights), \
            f"Z particles count ({len(z_masses)}) != weights count ({len(z_new_weights)})"
        print(f"Found {len(z_masses)} Z bosons, weight array matched (sum Z per event: {sum_z_count})")

        # Define 2D plot configurations for Z boson kinematics
        plot_2d_configs = [
            (z_pt, z_eta, "pT (GeV)", "Eta (η)", "pt_vs_eta"),
            (z_pt, z_phi, "pT (GeV)", "Phi (φ)", "pt_vs_phi"),
            (z_pz, z_eta, "pZ (GeV)", "Eta (η)", "pz_vs_eta"),
            (z_pz, z_phi, "pZ (GeV)", "Phi (φ)", "pz_vs_phi"),
            (z_eta, z_phi, "Eta (η)", "Phi (φ)", "eta_vs_phi"),
            (z_pt, z_masses, "pT (GeV)", "Mass (GeV)", "pt_vs_mass"),
            (z_eta, z_masses, "Eta (η)", "Mass (GeV)", "eta_vs_mass"),
            (z_pz, z_pt, "pZ (GeV)", "pT (GeV)", "pz_vs_pt"),
            (z_energy, z_pt, "Energy (GeV)", "pT (GeV)", "energy_vs_pt")
        ]

        # Generate 2D plots (non-weighted and weighted)
        for x_data, y_data, xlabel, ylabel, plot_name in plot_2d_configs:
            # Non-weighted 2D plot
            plot_2d_distribution(
                x_data=x_data,
                y_data=y_data,
                weights=None,
                title=f"Z Boson {xlabel} vs {ylabel} (Non-weighted)",
                xlabel=xlabel,
                ylabel=ylabel,
                filename=f"{output_dir}/z_{plot_name}_2d_nonweighted.png",
                bins=(50, 50),
                range=None
            )
            # Weighted 2D plot
            plot_2d_distribution(
                x_data=x_data,
                y_data=y_data,
                weights=z_new_weights,
                title=f"Z Boson {xlabel} vs {ylabel} (Weighted: 1+NC/SM)",
                xlabel=xlabel,
                ylabel=ylabel,
                filename=f"{output_dir}/z_{plot_name}_2d_weighted.png",
                bins=(50, 50),
                range=None
            )
        print("All 2D distribution plots (pZ/pT vs Eta/Phi + additional) generated")

        # Define Z boson variables for 1D comparison plots
        z_vars = [
            ("Mass", z_masses, "Mass (GeV)"),
            ("PT", z_pt, "pT (GeV)"),
            ("Eta", z_eta, "Eta (η)"),
            ("Phi", z_phi, "Phi (φ)"),
            ("Energy", z_energy, "Energy (GeV)"),
            ("Pz", z_pz, "Pz (GeV)")
        ]
        all_z_mask = np.ones_like(z_new_weights, dtype=bool)
        print(f"Total Z bosons: {np.sum(all_z_mask)}")

        # Generate 1D comparison plots for all Z boson variables
        for var_name, var_data, xlabel in z_vars:
            plot_weighted_comparison(
                original_values=var_data[all_z_mask],
                weighted_values=var_data[all_z_mask],
                weights=z_new_weights[all_z_mask],
                title=f"Z Boson {var_name} Comparison (All Events)",
                xlabel=xlabel,
                filename=f"{output_dir}/z_{var_name.lower()}_comparison_all_events.png",
                bins=100,
                range=None,
                label1="SM (Unweighted)",
                label2="Weighted (1+NC/SM)",
                color1='darkblue',
                color2='crimson',
                normalize=False
            )
        print("All Z boson comparison plots (all events) generated")

        # Generate plots for Z bosons with weight magnitude filters
        weight_abs_less50_mask = np.abs(z_new_weights) < 50
        weight_abs_more50_mask = np.abs(z_new_weights) > 50
        print(f"Z bosons with |Weight| < 50: {np.sum(weight_abs_less50_mask)}")
        print(f"Z bosons with |Weight| > 50: {np.sum(weight_abs_more50_mask)}")

        for var_name, var_data, xlabel in z_vars:
            # Plot for |Weight| < 50
            if np.sum(weight_abs_less50_mask) > 0:
                plot_weighted_comparison(
                    original_values=var_data[weight_abs_less50_mask],
                    weighted_values=var_data[weight_abs_less50_mask],
                    weights=z_new_weights[weight_abs_less50_mask],
                    title=f"Z Boson {var_name} Comparison (|Weight| < 50)",
                    xlabel=xlabel,
                    filename=f"{output_dir}/z_{var_name.lower()}_comparison_abs_weight_less50.png",
                    bins=100,
                    range=None,
                    label1="SM (Unweighted)",
                    label2="Weighted (1+NC/SM)",
                    color1='darkblue',
                    color2='crimson',
                    normalize=False
                )
            # Plot for |Weight| > 50
            if np.sum(weight_abs_more50_mask) > 0:
                plot_weighted_comparison(
                    original_values=var_data[weight_abs_more50_mask],
                    weighted_values=var_data[weight_abs_more50_mask],
                    weights=z_new_weights[weight_abs_more50_mask],
                    title=f"Z Boson {var_name} Comparison (|Weight| > 50)",
                    xlabel=xlabel,
                    filename=f"{output_dir}/z_{var_name.lower()}_comparison_abs_weight_more50.png",
                    bins=100,
                    range=None,
                    label1="SM (Unweighted)",
                    label2="Weighted (1+NC/SM)",
                    color1='darkblue',
                    color2='crimson',
                    normalize=False
                )
        print("Z boson comparison plots (|Weight| < 50 and |Weight| > 50) generated")

        # Generate plots for small pT Z bosons with weight filters
        small_pt_mask = z_pt < 5
        huge_weight_mask = np.abs(z_new_weights) > 100
        small_pt_huge_weight_mask = small_pt_mask & huge_weight_mask
        normal_weight_mask = np.abs(z_new_weights) <= 100
        small_pt_normal_weight_mask = small_pt_mask & normal_weight_mask

        print("\n=== Small pT Event Statistics ===")
        print(f"Total Z bosons: {np.sum(all_z_mask)}")
        print(f"Small pT Z bosons (<5 GeV): {np.sum(small_pt_mask)}")
        print(f"Small pT + huge weights (|weight|>100): {np.sum(small_pt_huge_weight_mask)}")
        print(f"Small pT + normal weights (|weight|<=100): {np.sum(small_pt_normal_weight_mask)}")

        # Define variables for peak check (Phi/Eta)
        peak_check_vars = [
            ("Phi", z_phi, "Phi (φ)"),
            ("Eta", z_eta, "Eta (η)")
        ]

        # Generate small pT plots
        for var_name, var_data, xlabel in peak_check_vars:
            # Small pT + huge weights
            if np.sum(small_pt_huge_weight_mask) > 0:
                plot_weighted_comparison(
                    original_values=var_data[small_pt_huge_weight_mask],
                    weighted_values=var_data[small_pt_huge_weight_mask],
                    weights=z_new_weights[small_pt_huge_weight_mask],
                    title=f"Z Boson {var_name} - Small pT (<5 GeV) + |Weight|>100",
                    xlabel=xlabel,
                    filename=f"{output_dir}/z_{var_name.lower()}_small_pt_huge_weights.png",
                    bins=100,
                    range=None,
                    label1="Unweighted (Small pT + |W|>100)",
                    label2="Weighted (Small pT + |W|>100)",
                    color1='orange',
                    color2='red',
                    normalize=False
                )

            # Small pT + normal weights
            if np.sum(small_pt_normal_weight_mask) > 0:
                plot_weighted_comparison(
                    original_values=var_data[small_pt_normal_weight_mask],
                    weighted_values=var_data[small_pt_normal_weight_mask],
                    weights=z_new_weights[small_pt_normal_weight_mask],
                    title=f"Z Boson {var_name} - Small pT (<5 GeV) + |Weight|<=100",
                    xlabel=xlabel,
                    filename=f"{output_dir}/z_{var_name.lower()}_small_pt_normal_weights.png",
                    bins=100,
                    range=None,
                    label1="Unweighted (Small pT + |W|<=100)",
                    label2="Weighted (Small pT + |W|<=100)",
                    color1='blue',
                    color2='darkblue',
                    normalize=False
                )

        print("Phi/Eta subset plots for small pT analysis generated")

        # Process extreme events (1 + NC/SM > 500)
        extreme_mask = (new_weights > 500) & valid_sm_mask
        extreme_indices = np.where(extreme_mask)[0]
        n_extreme_events = len(extreme_indices)
        
        print(f"\nFound {n_extreme_events} extreme events (1 + NC/SM > 500):")
        if n_extreme_events > 0:
            extreme_events = data[extreme_indices]
            extreme_weights = new_weights[extreme_indices]

            # Print extreme event details
            for idx, weight in zip(extreme_indices, extreme_weights):
                print(f"  Event {idx}: Weight = {weight:.4f}")

            # Generate extreme event weight distribution plot
            plot_weight_distribution(
                weights=extreme_weights,
                title="Weight Distribution of Extreme Events (1+NC/SM > 500)",
                filename=f"{output_dir}/extreme_events_weight_distribution.png",
                bins=50,
                color='red',
                range=None,
                normalize=False
            )
            print("Extreme events weight distribution plot generated")

            # Export extreme event data to PDF
            export_extreme_z_quark_momenta_to_pdf(extreme_events, extreme_weights, extreme_indices, output_dir)
        else:
            print("No extreme events found (1 + NC/SM > 500)")

        print(f"\nAll tasks completed! Plots and PDF saved to: {output_dir}")

    except Exception as e:
        print(f"\nError during processing: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    # Command line argument validation
    if len(sys.argv) != 3:
        print("Usage: python3 draw_nc_sm_graph.py <input_file> <tree_name>")
        print("Example: python3 draw_nc_sm_graph.py ./result/update_MG_315200_1e-4_weighted_events.root LHEF")
        sys.exit(1)

    input_file = sys.argv[1]
    tree_name = sys.argv[2]
    
    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' does not exist")
        sys.exit(1)

    # Run main processing function
    main(input_file, tree_name)