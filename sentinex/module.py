from collections import defaultdict
from typing import Optional

from equinox import filter_jit
from jax import tree_util
from termcolor import colored

__all__ = ["Module"]

@tree_util.register_pytree_node_class
class Module:
    """Base class for all ``sentinex`` Modules.

    Most layers and models be a direct/indirect subclass of this class.
    Other subclasses may provide more refined control and predefined methods, 
    so please check them out as well.

    A module class consists of all attributes that make a class compatible with the ``sentinex`` components.

    Args:
      name (str, optional): The name of the instance. Defaults to ``Module``.
    """
    
    _set_name: dict = defaultdict(int) # Just a name setting library
    _annotation_dict: dict = defaultdict(dict)
    
    def __init__(self, name: Optional[str] = "Module") -> None:
        self.name: str = f"{name}_{Module._set_name[name]}"; Module._set_name[name] += 1
        
        # Creating instance copy of annotations:
        self._annotations = self._create_annotations()
    
    def __init_subclass__(cls) -> None:
        """All subclasses get registered as PyTrees"""
        tree_util.register_pytree_node_class(cls)
    
    @classmethod
    def _create_annotations(cls):
        return cls.__annotations__.copy()
    
    @property
    def annotations(self):
        """Returns all the annotations defined in the class."""
        return self._annotations
    
    def set_annotation(self, annotation_name, annotation_type):
        """Sets an annotation after class definition."""
        self._annotations[annotation_name] = annotation_type
    
    def del_annotation(self, annotation_name):
        """Returns and deletes an annotation after class definition."""
        return self._annotations.pop(annotation_name)
    
    def _fun(self, x):
        if isinstance(x[0], Module):
            return x[0].dynamic_attributes()
        else:
            return x[0]
    
    def dynamic_attributes(self):
        """Returns all the attributes that are marked as dynamic."""
        return list(map(self._fun, self.tree_flatten()[0]))
        
    def compile(self):
        self.__call__ = filter_jit(self.__call__)
    
    # Jax Tree Methods:
    def tree_flatten(self):
        instance_dict = vars(self).copy()
        leaves = []
        def flatten_recipe(x):
            if x in instance_dict:
                leaves.append(instance_dict.pop(x))
            else:
                print(colored(f"""Warning from {self.name}. 
                              All type annotated datas should be defined as an attribute. Undefined Annotation: {x}""", "red"))
                
        list(map(flatten_recipe, self.annotations)) # type: ignore        
                
        aux_data = instance_dict.copy()
        Module._annotation_dict[self.name] = self.annotations
        return leaves, aux_data
    
    @classmethod
    def tree_unflatten(cls, aux_data: dict, leaves: list):
        instance = cls.__new__(cls)
        leaves_dict = dict(zip(Module._annotation_dict[aux_data['name']], leaves))
        vars(instance).update(leaves_dict)
        vars(instance).update(aux_data)
        return instance

    def __repr__(self) -> str:
        """Displays the basic structure of a Module class."""
        return f"{vars(self)}"

# Fixing inspections:
Module.__module__ = "sentinex"
