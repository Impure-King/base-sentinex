from collections import defaultdict
from typing import Optional, Dict, DefaultDict, Self, Tuple, Any

from equinox import filter_jit
from jax import tree_util
from termcolor import colored
from jax import Array, tree_map

__all__ = ["Module"]

def formatter(attribute_vars, annotation_vars):
    repr_str = ""
    for key in annotation_vars:
        repr_str += f"{key}: {attribute_vars[key]}\n"
    return repr_str.split('\n')
        
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
    
    _set_name: DefaultDict = defaultdict(int) # Just a name setting dictionary.
    _annotation_dict: DefaultDict = defaultdict(dict) # Holds annotations in between flattening and unflattening.
    
    def __init__(self, 
                 name: str = "Module", 
                 dynamic: bool = False,
                 trainable: bool = True) -> None:
        self.name: str = f"{name}_{Module._set_name[name]}"; Module._set_name[name] += 1
        
        # Creating instance copy of annotations:
        self._annotations: dict = self._create_annotations()
        
        
        # Jitting __call__ method:
        self.dynamic: bool = dynamic
        if hasattr(self, "__call__") and dynamic:
            self.__call__ = filter_jit(self.__call__)
        
        self.trainable = trainable

        self.depth = 0

        
    def __init_subclass__(cls) -> None:
        """All subclasses get registered as PyTrees"""
        tree_util.register_pytree_node_class(cls)
    
    @classmethod
    def _create_annotations(cls) -> Dict[str, type]:
        """Creates the instance version of annotations and fields.
        Only for use in ``Module.__init__``.

        Returns:
            dict: The copy of the class's annotation dictionary.
        """
        return cls.__annotations__.copy()
    
    @property
    def annotations(self) -> Dict[str, type]:
        """Return's the instance's locally defined annotations.
        Changing instance based annotation's doesn't affect class.

        Returns:
            dict: The instance annotation dictionary.
        """
        return self._annotations
    
    def set_annotation(self, annotation_name: str, annotation_type: type) -> None:
        """Modifies the instance annotation, by setting a annotation,
        when given annotation_name and annotation_type.

        Args:
            annotation_name (str): The name of the attribute to mark dynamic.
            annotation_type (type): The type of the specified attribute.
        """
        self._annotations[annotation_name] = annotation_type
    
    def del_annotation(self, annotation_name: str) -> type:
        """Deletes the specified instance annotation and returns it's type.

        Args:
            annotation_name (str): The name of the attribute to mark static.

        Returns:
            _type_: _description_
        """
        return self._annotations.pop(annotation_name)
    
    def _fun(self, x):
        if isinstance(x[0], Module):
            return x[0].dynamic_attributes()
        else:
            return x[0]
    
    def dynamic_attributes(self) -> list:
        """Returns the instance's dynamic attributes.

        Returns:
            list: The list of dynamic attributes.
        """
        return list(map(self._fun, self.tree_flatten()[0]))
    
    # Jax Tree Methods:
    def tree_flatten(self) -> Tuple[list, Dict[str, Any]]:
        """Flattens the instance's dynamic and static attributes
        for processing with appropriate jax transformations.

        Returns:
            tuple: A tuple of leaves and auxiliary data.
        """
        instance_dict: dict = vars(self).copy()
        leaves: list = []
        def flatten_recipe(x):
            if x in instance_dict:
                leaves.append(instance_dict.pop(x))
            else:
                print(colored(f"""Warning from {self.name}. 
                              All type annotated datas should be defined as an attribute. Undefined Annotation: {x}""", "red"))
                
        list(map(flatten_recipe, self.annotations)) # type: ignore        
                
        aux_data: dict = instance_dict.copy()
        Module._annotation_dict[self.name] = self.annotations
        return leaves, aux_data
    
    @classmethod
    def tree_unflatten(cls, aux_data: Dict[str, Any], leaves: list) -> Self:
        """Unflattens and returns a Module instance, when given the appropriate
        auxiliary data and leaves.

        Args:
            aux_data (dict): The static data from flattening.
            leaves (list): The transformed dynamic data.

        Returns:
            Self: An unflattened instance that possess updates attributes from the given data.
        """
        instance: Self = cls.__new__(cls)
        leaves_dict: dict = dict(zip(Module._annotation_dict[aux_data['name']], leaves))
        vars(instance).update(leaves_dict)
        vars(instance).update(aux_data)
        return instance
    
    def get_attr(self, key: str) -> Any:
        """Returns an attribute and prevents an AttributeError.

        Args:
            key (str): The attribute key.

        Returns:
            Any: The attribute requested.
        """
        try:
            return self.__getattribute__(key)
        except AttributeError:
            return "Not Defined"

    @property
    def trainable_variables(self):
        trainable_list = []
        def filter_array(x):
            if isinstance(x, Array):
                trainable_list.append(x)
        tree_map(filter_array, self)
        return trainable_list
    
    def __repr__(self) -> str:
        """Displays the name of the module."""
        if len(list(self.annotations.values())):
            repr_str = "\n" * self.depth + f"{self.name}:"
            for key in self.annotations:
                value = self.get_attr(key)
                if isinstance(value, Array):
                    repr_str += "\n" * (self.depth + 1) + f"{value.shape}"
                elif isinstance(value, Module):
                    value.depth = self.depth + 1
                    repr_str += f"{value}"
                else:
                    repr_str += "\n" * (self.depth + 1) + f"{value!r}"
        else:
            repr_str = "\n" * self.depth + self.name
        return repr_str

# Fixing inspections:
Module.__module__ = "sentinex"
