import torch

from app.services.context_adapter_model import ContextAdapter


def test_context_adapter_output_shape():
    model = ContextAdapter(
        context_dim=384,
        hidden_dim=1024,
    )

    encoder_hidden = torch.randn(
        2,
        16,
        1024,
    )

    context_embedding = torch.randn(
        2,
        384,
    )

    output = model(
        encoder_hidden,
        context_embedding,
    )

    assert output.shape == (
        2,
        16,
        1024,
    )


def test_context_adapter_projects_context():
    model = ContextAdapter(
        context_dim=384,
        hidden_dim=1024,
    )

    context = torch.randn(
        4,
        384,
    )

    projected = model.context_projection(
        context
    )

    assert projected.shape == (
        4,
        1024,
    )


def test_context_adapter_is_trainable():
    model = ContextAdapter(
        context_dim=384,
        hidden_dim=1024,
    )

    trainable_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]

    assert trainable_parameters


def test_context_adapter_batch_independence():
    model = ContextAdapter(
        context_dim=384,
        hidden_dim=1024,
    )

    encoder_hidden = torch.randn(
        2,
        8,
        1024,
    )

    context = torch.randn(
        2,
        384,
    )

    output = model(
        encoder_hidden,
        context,
    )

    assert output.shape[0] == 2
    assert output.shape[1] == 8
    assert output.shape[2] == 1024