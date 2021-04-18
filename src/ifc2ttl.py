import ifcopenshell
from owlready2 import *


class Mappping_dict():
    def __init__(self):
        pass

    def ifc_map_fireonto(self):
        if not hasattr(self, 'ifc_map_fireonto_dict'):
            # if 'Todo' is in the value the inverse function (such as fireonto_map_ifc) will not inverse the dict item
            self.ifc_map_fireonto_dict = {
                # class
                'IfcBuildingStorey': 'BuildingStorey',
                'IfcBuilding': 'BuildingRegion',
                'IfcSpace': 'BuildingSpace',
                'IfcColumn': 'Column',
                'IfcOpeningElement': 'Holes_Todo',
                'IfcBeam': 'Beam',
                'IfcSlab': 'Floor',
                'IfcWallStandardCase': 'Wall',
                'IfcWall': 'Wall',
                'IfcWindow': 'Window',
                'IfcDoor': 'Doors',
                'IfcStair': 'Stairs',
                'IfcMember': '?_Todo',
                'IfcRailing': '?_Todo',
                'IfcStairFlight': '?_Todo',
                # data property
                '耐火极限': 'hasFireResistanceLimits_hour',
                '是否承重': 'isLoadBearing_Boolean',
                '是否是安全出口': 'IsSecurityExits_Boolean',
                '是否是防火墙': 'isFireWall_Boolean',
                '是否为防火分区': 'isFireProtectionSubdivision_Boolean',
                '容纳人数': 'hasMaxNumberOfHuman',
                'GlobalId': 'hasGlobalId',
                '面积': 'hasBuildingArea_m2',
            }
        return self.ifc_map_fireonto_dict

    # this is the inverse function of ifc_map_fireonto
    def fireonto_map_ifc(self):
        if not hasattr(self, 'fireonto_map_ifc_dict'):
            self.fireonto_map_ifc_dict = {value: key for key, value in self.ifc_map_fireonto_dict.items() if
                                          'Todo' not in value}
        return self.fireonto_map_ifc_dict


class Building_element():
    def __init__(self, Type, GlobalId, Name):
        self.element_type = Type  # IfcColumn
        self.element_id = GlobalId  # '0gKFhH4357pRQxVB7vb_0Y'
        self.element_name = Name  # 普通墙体 500x300
        self.data_properties = []  # list of tuple, each tuple is a (data_property_name, property_value)
        self.obj_properties = []  # list of tuple, each tuple is a (obj_property_name, range_element)
        # self.instance_name IfcColumn1
        # self.element_class is the class name in the ontology

    def add_dataprop(self, property_name, property_value):
        self.data_properties.append((property_name, property_value))

    '''
    range_element is a object of the Building_element class
    '''

    def add_objprop(self, property_name, range_element):
        self.obj_properties.append((property_name, range_element))

    def write_dataprop_ttl(self, mapping_dict, number_item: int, prefix='firecodes:'):
        # only test for 'hasFireResistanceLimits(hour)'
        # the prefix short for BuildingDesignFireCodesOntology is firecodes:
        # set class
        # self.instance_name is used for object property generation
        self.instance_name = self.element_type + str(number_item)
        ttl_content = ''
        if self.element_type in mapping_dict:
            # element_class is the class name in the ontology
            self.element_class = mapping_dict[self.element_type]
            # if 'Todo' in element_class the ontology do not have the class, and will raise error
            if 'Todo' not in self.element_class:
                ttl_content = prefix + self.instance_name + ' rdf:type owl:NamedIndividual , ' + '<http://www.semanticweb.org/16424/ontologies/2020/10/untitled-ontology-8#' + self.element_class + '> .\n'
                # set global id
                ttl_content += prefix + self.instance_name + ' ' + prefix + mapping_dict[
                    'GlobalId'] + ' ' + '"' + self.element_id + '"^^xsd:string' + ' .\n'
                # set data data_properties
                for property in self.data_properties:
                    if property[0] in mapping_dict:
                        dataproperty_name = mapping_dict[property[0]]
                        if type(property[1]) is bool:
                            ttl_content += prefix + self.instance_name + ' ' + prefix + dataproperty_name + ' ' + '"' + str(
                                property[1]).lower() + '"^^xsd:boolean' + ' .\n'
                        elif type(property[1]) is float:
                            ttl_content += prefix + self.instance_name + ' ' + prefix + dataproperty_name + ' ' + '"' + str(
                                property[1]) + '"^^xsd:decimal' + ' .\n'
                    else:
                        continue
        return ttl_content

    def write_objprop_ttl(self):
        ttl_content = ''
        for obj_property in self.obj_properties:
            ttl_content += ':' + self.instance_name + ' :' + obj_property[0] + ' :' + obj_property[
                1].instance_name + ' .\n'
        return ttl_content

    '''
    State: Use
    Function: get one single element Type, GlobaId, Name, necessary dataproperty from ifc file
    Todo: Now only the property in the property_set_name='消防系统' is considered, the area dataproperty is not considered
    '''

    @staticmethod
    # get element '消防系统' sub data_properties for every element
    # TODO 全转换
    def get_single_element_prop(ifc_obj, property_name='', property_set_names=['消防系统']):
        Type = ifc_obj.is_a()
        GlobaId = ifc_obj.GlobalId
        Name = ifc_obj.Name
        element = Building_element(Type, GlobaId, Name)

        # for test
        # if ifc_obj.is_a('ifcSlab'):
        #     print('ok')
        for property_set_name in property_set_names:
            # family property
            for family in ifc_obj.IsTypedBy:
                if family.is_a('IfcRelDefinesByType'):
                    property_fsets = family.RelatingType.HasPropertySets
                    if property_fsets is not None:
                        for property_fset in property_fsets:
                            if property_fset.Name == property_set_name:
                                for property in property_fset.HasProperties:
                                    element.add_dataprop(property.Name, property.NominalValue.wrappedValue)
                            else:
                                continue
                    else:
                        continue
            # instance property
            for definition in ifc_obj.IsDefinedBy:
                if not definition.is_a('IfcRelDefinesByProperties'):
                    continue
                property_set = definition.RelatingPropertyDefinition
                if property_set.Name == property_set_name or property_set.Name == property_set_name + "(Type)":
                    if property_set.is_a('IfcElementQuantity'):
                        for quantity in property_set.Quantities:
                            if quantity.is_a('IfcQuantityArea'):
                                if quantity.Name == property_name:
                                    return quantity.AreaValue
                            elif quantity.is_a('IfcQuantityVolume'):
                                if quantity.Name == property_name:
                                    return quantity.VolumeValue
                            else:
                                continue  # there are more types

                    elif property_set.is_a('IfcPropertySet'):
                        for property in property_set.HasProperties:
                            if property.is_a('IfcPropertySingleValue'):
                                element.add_dataprop(property.Name, property.NominalValue.wrappedValue)
                                # if property.Name == '面积':
                                #     print(property.Name)
                                # return property.NominalValue.wrappedValue if property.NominalValue else None
                            else:
                                continue  # there are more types
                    else:
                        continue  # there are more types
        return element

    '''
    State: Use
    Function: get the obj_prop between two element from the elements and ifc file
    Todo: now only consider two type of obj_prop
        1. IfcBuilding hasBuildingSpatialElement BuildingStorey
        2. IfcSpace hasBuildingElement xxx
    '''

    @staticmethod
    def get_elements_objprop(proceed_elements, ifc_products):
        def search_element_by_GlobalId(proceed_elements, GlobalId):
            for element in proceed_elements:
                if element.element_id == GlobalId:
                    return element

        for ifc_obj in ifc_products:
            domain_type = ifc_obj.is_a()
            if domain_type == 'IfcBuilding':
                domain_GlobaId = ifc_obj.GlobalId
                domain_element = search_element_by_GlobalId(proceed_elements, domain_GlobaId)
                if len(ifc_obj.IsDecomposedBy) > 0:
                    range_objs = ifc_obj.IsDecomposedBy[0].RelatedObjects
                    '''IfcBuilding hasBuildingSpatialElement BuildingStorey'''
                    for range_obj in range_objs:
                        range_name = range_obj.Name
                        '''
                        The ifcstorey is not the real storey in the world, judge whether it is a real floor through keyword matching method
                        '''
                        if '地面' in range_name or '楼' in range_name:
                            range_type = range_obj.is_a()
                            range_GlobaId = range_obj.GlobalId
                            range_element = search_element_by_GlobalId(proceed_elements, range_GlobaId)
                            # obj_prop_tag = Building_element.get_objprop(domain_type, range_type)
                            # assert obj_prop_tag == 'hasBuildingSpatialElement', 'The obj_prop_tag is not True'
                            obj_prop_tag = 'hasBuildingSpatialElement'
                            domain_element.add_objprop(obj_prop_tag, range_element)
            elif domain_type == 'IfcSpace':
                domain_GlobaId = ifc_obj.GlobalId
                domain_element = search_element_by_GlobalId(proceed_elements, domain_GlobaId)
                for bounding_obj in ifc_obj.BoundedBy:
                    range_GlobaId = bounding_obj.RelatedBuildingElement.GlobalId
                    range_element = search_element_by_GlobalId(proceed_elements, range_GlobaId)
                    range_type = bounding_obj.RelatedBuildingElement.is_a()
                    # obj_prop_tag = Building_element.get_objprop(domain_type, range_type)
                    # assert obj_prop_tag == 'hasBuildingElement', 'The obj_prop_tag is not True'
                    obj_prop_tag = 'hasBuildingElement'
                    domain_element.add_objprop(obj_prop_tag, range_element)

    '''
    State: use
    Function: get the most proper obj_prop by domain class type(string) and range class type(string), Now just two types (hasBuildingSpatialElement) and hasBuildingElement has been considered
    '''

    @staticmethod
    def get_objprop(domain: str, range: str,
                    owl_file=r"D:\OntologyExample\Owlready2\BuildingDesignFireCodesOntology.owl"):
        onto = get_ontology(owl_file).load()

        def ontoclass_tostr(ontoclass):
            return str(ontoclass).split('.')[1]

        def get_ontoclass(ontology, class_name: str):
            for oneclass in ontology.classes():
                if str(oneclass).split('.')[1] == class_name:
                    return oneclass

        domain_class = get_ontoclass(onto, domain)
        range_class = get_ontoclass(onto, range)
        if 'BuildingSpatialElement' in list(map(ontoclass_tostr, domain_class.ancestors())):
            if 'BuildingSpatialElement' in list(map(ontoclass_tostr, range_class.ancestors())):
                return 'hasBuildingSpatialElement'
            elif 'BuildingComponentElement' in list(map(ontoclass_tostr, range_class.ancestors())):
                return 'hasBuildingElement'
            else:
                return None
        else:
            return None


def gen_ttl_file(ifc_file='./data/ifcfile/Plant_ByhandV2.ifc', ttl_file='./data/ttlfile/Plant_instance.ttl'):
    def get_allprop_from_ifc(ifcfile='./data/ifcfile/Plant_ByhandV3.ifc'):
        ifc_file = ifcopenshell.open(ifcfile)
        products = ifc_file.by_type('IfcProduct')
        elements = []
        for product in products:
            elements.append(Building_element.get_single_element_prop(product, property_set_names=['消防系统', '尺寸标注']))
        return elements

    proceed_elements = get_allprop_from_ifc(ifc_file)
    content = """@prefix : <http://www.semanticweb.org/16424/ontologies/2020/10/BuildingDesignFireCodesOntology#> .
@prefix owl: <http://www.w3.org/2002/07/owl#> .
@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
@prefix xml: <http://www.w3.org/XML/1998/namespace> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@base <http://www.semanticweb.org/16424/ontologies/2020/10/BuildingDesignFireCodesOntology> .

#################################################################
#    Individuals
#################################################################
###  http://www.semanticweb.org/16424/ontologies/2020/10/BuildingDesignFireCodesOntology#column1_test\n\n"""

    my_prefix_short = ':'

    ifc_map_fireonto_dict = Mappping_dict().ifc_map_fireonto()
    for index, element in enumerate(proceed_elements):
        element_ttl_content = element.write_dataprop_ttl(ifc_map_fireonto_dict, index, prefix=my_prefix_short)
        content += element_ttl_content

    # get objpropery ttl file
    ifc_file = ifcopenshell.open(ifc_file)
    ifc_products = ifc_file.by_type('IfcProduct')
    Building_element.get_elements_objprop(proceed_elements, ifc_products)
    for element in proceed_elements:
        content += element.write_objprop_ttl()
    with open(ttl_file, 'w', encoding='UTF-8') as ttl:
        ttl.write(content)


if __name__ == "__main__":
    gen_ttl_file(ifc_file='./data/ifcfile/Plant_ByhandV4.ifc')
    print('ok')
